#!/usr/bin/env python3
"""
Bib Tagger GUI - Modern Tkinter interface for bib number detection.
"""

import csv
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import (
    BooleanVar,
    Canvas,
    Frame,
    Label,
    StringVar,
    Tk,
    filedialog,
    messagebox,
    ttk,
)
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

from bib_tagger import BibTagger, get_image_files


# Color scheme - dark grey neutral with green accent
COLORS = {
    'bg': '#1a1a1a',           # Dark background
    'surface': '#2a2a2a',       # Card/surface background
    'surface_hover': '#3a3a3a', # Hover state
    'primary': '#4ade80',       # Primary accent (green)
    'primary_dark': '#22c55e',  # Primary hover (darker green)
    'success': '#4ade80',       # Success green
    'text': '#e5e5e5',          # Main text
    'text_dim': '#a3a3a3',      # Dimmed text
    'border': '#404040',        # Border color
}


def get_model_path() -> str:
    """Get the path to the ONNX model, handling bundled app case."""
    if getattr(sys, 'frozen', False):
        bundle_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(sys.executable).parent
        model_path = bundle_dir / 'models' / 'bib_detector.onnx'
        if model_path.exists():
            return str(model_path)

    script_dir = Path(__file__).parent
    model_path = script_dir / 'models' / 'bib_detector.onnx'
    if model_path.exists():
        return str(model_path)

    raise FileNotFoundError("Could not find bib_detector.onnx model")


def configure_styles():
    """Configure ttk styles for modern look."""
    style = ttk.Style()

    # Try to use clam theme as base (most customizable)
    try:
        style.theme_use('clam')
    except:
        pass

    # Configure colors
    style.configure('.',
        background=COLORS['bg'],
        foreground=COLORS['text'],
        fieldbackground=COLORS['surface'],
        borderwidth=0,
        focusthickness=0,
    )

    # Frame style
    style.configure('TFrame', background=COLORS['bg'])
    style.configure('Card.TFrame', background=COLORS['surface'])

    # Label styles
    style.configure('TLabel',
        background=COLORS['bg'],
        foreground=COLORS['text'],
        font=('Segoe UI', 11)
    )
    style.configure('Title.TLabel',
        background=COLORS['bg'],
        foreground=COLORS['text'],
        font=('Segoe UI', 28, 'bold')
    )
    style.configure('Subtitle.TLabel',
        background=COLORS['bg'],
        foreground=COLORS['text_dim'],
        font=('Segoe UI', 11)
    )
    style.configure('Card.TLabel',
        background=COLORS['surface'],
        foreground=COLORS['text'],
        font=('Segoe UI', 11)
    )
    style.configure('CardTitle.TLabel',
        background=COLORS['surface'],
        foreground=COLORS['text'],
        font=('Segoe UI', 12, 'bold')
    )
    style.configure('Status.TLabel',
        background=COLORS['bg'],
        foreground=COLORS['text_dim'],
        font=('Segoe UI', 10)
    )

    # Entry style
    style.configure('TEntry',
        fieldbackground=COLORS['surface'],
        foreground=COLORS['text'],
        insertcolor=COLORS['text'],
        borderwidth=0,
        padding=(12, 10),
    )
    style.map('TEntry',
        fieldbackground=[('focus', COLORS['surface_hover'])],
    )

    # Button styles
    style.configure('TButton',
        background=COLORS['surface'],
        foreground=COLORS['text'],
        padding=(20, 12),
        font=('Segoe UI', 11),
        borderwidth=0,
    )
    style.map('TButton',
        background=[('active', COLORS['surface_hover']), ('pressed', COLORS['surface_hover'])],
    )

    style.configure('Primary.TButton',
        background=COLORS['primary'],
        foreground='#11111b',
        padding=(30, 14),
        font=('Segoe UI', 12, 'bold'),
    )
    style.map('Primary.TButton',
        background=[('active', COLORS['primary_dark']), ('pressed', COLORS['primary_dark']), ('disabled', COLORS['surface'])],
        foreground=[('disabled', COLORS['text_dim'])],
    )

    # Checkbutton style
    style.configure('TCheckbutton',
        background=COLORS['surface'],
        foreground=COLORS['text'],
        font=('Segoe UI', 11),
        padding=(8, 6),
    )
    style.map('TCheckbutton',
        background=[('active', COLORS['surface'])],
    )

    # Progressbar style
    style.configure('TProgressbar',
        background=COLORS['primary'],
        troughcolor=COLORS['surface'],
        borderwidth=0,
        thickness=6,
    )

    # Separator
    style.configure('TSeparator', background=COLORS['border'])


class BibTaggerApp:
    """Main GUI application for Bib Tagger."""

    def __init__(self, root: Tk = None):
        self.root = root if root else Tk()
        self.root.title("Bib Tagger")
        self.root.geometry("1400x800")
        self.root.minsize(1100, 700)
        self.root.configure(bg=COLORS['bg'])

        # Configure styles
        configure_styles()

        # State
        self.folder_path = StringVar()
        self.generate_csv = BooleanVar(value=True)
        self.show_preview = BooleanVar(value=True)
        self.save_debug_images = BooleanVar(value=False)
        self.write_metadata = BooleanVar(value=False)
        self.include_subfolders = BooleanVar(value=False)
        self.processing = False
        self.cancel_requested = False
        self.tagger = None
        self.current_preview_image = None  # Keep reference to prevent garbage collection

        # Image navigation state
        self.debug_image_list = []  # List of (path, filename) tuples
        self.current_image_index = 0

        self._create_widgets()

        # Update image count when subfolder option changes
        self.include_subfolders.trace_add('write', lambda *args: self._update_image_count())

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with two columns
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Configure grid weights for responsive layout
        main_frame.columnconfigure(0, weight=1, minsize=450)  # Left panel (controls)
        main_frame.columnconfigure(1, weight=2, minsize=500)  # Right panel (preview)
        main_frame.rowconfigure(0, weight=1)

        # ===== LEFT PANEL (Controls & Log) =====
        left_panel = ttk.Frame(main_frame, style='TFrame')
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 15))

        # Header section
        header_frame = ttk.Frame(left_panel, style='TFrame')
        header_frame.pack(fill='x', pady=(0, 20))

        ttk.Label(
            header_frame,
            text="Bib Tagger",
            style='Title.TLabel'
        ).pack(anchor='w')

        ttk.Label(
            header_frame,
            text="Detect and read bib numbers from race photos",
            style='Subtitle.TLabel'
        ).pack(anchor='w', pady=(4, 0))

        # Folder selection card
        folder_card = ttk.Frame(left_panel, style='Card.TFrame', padding=15)
        folder_card.pack(fill='x', pady=(0, 10))

        ttk.Label(
            folder_card,
            text="Image Folder",
            style='CardTitle.TLabel'
        ).pack(anchor='w', pady=(0, 8))

        folder_input_frame = ttk.Frame(folder_card, style='Card.TFrame')
        folder_input_frame.pack(fill='x')

        self.folder_entry = ttk.Entry(
            folder_input_frame,
            textvariable=self.folder_path,
            font=('Segoe UI', 10),
        )
        self.folder_entry.pack(side='left', fill='x', expand=True, padx=(0, 8), ipady=3)

        browse_btn = ttk.Button(
            folder_input_frame,
            text="Browse",
            command=self._browse_folder,
            style='TButton'
        )
        browse_btn.pack(side='right')

        # Options card
        options_card = ttk.Frame(left_panel, style='Card.TFrame', padding=15)
        options_card.pack(fill='x', pady=(0, 10))

        ttk.Label(
            options_card,
            text="Output Options",
            style='CardTitle.TLabel'
        ).pack(anchor='w', pady=(0, 8))

        self.csv_check = ttk.Checkbutton(
            options_card,
            text="Generate CSV report",
            variable=self.generate_csv,
            style='TCheckbutton'
        )
        self.csv_check.pack(anchor='w', pady=(0, 2))

        self.preview_check = ttk.Checkbutton(
            options_card,
            text="Show detection preview",
            variable=self.show_preview,
            style='TCheckbutton'
        )
        self.preview_check.pack(anchor='w')

        self.save_debug_check = ttk.Checkbutton(
            options_card,
            text="Save debug images to disk",
            variable=self.save_debug_images,
            style='TCheckbutton'
        )
        self.save_debug_check.pack(anchor='w', pady=(2, 0))

        self.metadata_check = ttk.Checkbutton(
            options_card,
            text="Write bib numbers to image metadata (IPTC)",
            variable=self.write_metadata,
            style='TCheckbutton'
        )
        self.metadata_check.pack(anchor='w', pady=(2, 0))

        self.subfolders_check = ttk.Checkbutton(
            options_card,
            text="Include subfolders",
            variable=self.include_subfolders,
            style='TCheckbutton'
        )
        self.subfolders_check.pack(anchor='w', pady=(2, 0))

        # Button frame for Start, Cancel, and Open Folder buttons
        button_frame = ttk.Frame(left_panel, style='TFrame')
        button_frame.pack(pady=(8, 15))

        # Start button
        self.start_btn = ttk.Button(
            button_frame,
            text="Start Processing",
            command=self._start_processing,
            style='Primary.TButton'
        )
        self.start_btn.pack(side='left', padx=(0, 10))

        # Cancel button (initially hidden)
        self.cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel_processing,
            style='TButton'
        )
        # Don't pack yet - will be shown during processing

        # Open Folder button (initially hidden)
        self.open_folder_btn = ttk.Button(
            button_frame,
            text="Open Folder",
            command=self._open_folder,
            style='TButton'
        )
        # Don't pack yet - will be shown after processing

        # Progress section
        progress_frame = ttk.Frame(left_panel, style='TFrame')
        progress_frame.pack(fill='x', pady=(0, 10))

        self.progress_label = ttk.Label(
            progress_frame,
            text="Ready to process",
            style='TLabel'
        )
        self.progress_label.pack(anchor='w', pady=(0, 6))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            style='TProgressbar'
        )
        self.progress_bar.pack(fill='x')

        # Log section
        log_frame = ttk.Frame(left_panel, style='TFrame')
        log_frame.pack(fill='both', expand=True)

        ttk.Label(
            log_frame,
            text="Processing Log",
            style='TLabel'
        ).pack(anchor='w', pady=(0, 6))

        # Log text with custom styling
        self.log_text = ScrolledText(
            log_frame,
            height=10,
            font=('JetBrains Mono', 9) if sys.platform != 'darwin' else ('Monaco', 9),
            bg=COLORS['surface'],
            fg=COLORS['text'],
            insertbackground=COLORS['text'],
            selectbackground=COLORS['primary'],
            selectforeground='#11111b',
            borderwidth=0,
            highlightthickness=0,
            padx=10,
            pady=10,
            state='disabled'
        )
        self.log_text.pack(fill='both', expand=True)

        # Status bar
        status_frame = ttk.Frame(left_panel, style='TFrame')
        status_frame.pack(fill='x', pady=(10, 0))

        self.status_label = ttk.Label(
            status_frame,
            text="Select a folder containing race photos to begin",
            style='Status.TLabel'
        )
        self.status_label.pack(anchor='w')

        # ===== RIGHT PANEL (Image Preview) =====
        right_panel = ttk.Frame(main_frame, style='Card.TFrame', padding=15)
        right_panel.grid(row=0, column=1, sticky='nsew')

        # Preview header
        preview_header = ttk.Frame(right_panel, style='Card.TFrame')
        preview_header.pack(fill='x', pady=(0, 10))

        ttk.Label(
            preview_header,
            text="Detection Preview",
            style='CardTitle.TLabel'
        ).pack(side='left')

        self.preview_filename_label = ttk.Label(
            preview_header,
            text="",
            style='Card.TLabel'
        )
        self.preview_filename_label.pack(side='right')

        # Image canvas container (for centering)
        self.preview_container = Frame(right_panel, bg=COLORS['surface'])
        self.preview_container.pack(fill='both', expand=True)

        # Canvas for image display
        self.preview_canvas = Canvas(
            self.preview_container,
            bg=COLORS['surface'],
            highlightthickness=0
        )
        self.preview_canvas.pack(fill='both', expand=True)

        # Placeholder text
        self.preview_canvas.create_text(
            0, 0,
            text="Debug images will appear here\nas processing runs",
            fill=COLORS['text_dim'],
            font=('Segoe UI', 12),
            tags='placeholder',
            justify='center'
        )

        # Bind resize event to reposition placeholder and resize image
        self.preview_canvas.bind('<Configure>', self._on_preview_resize)

        # Navigation controls
        nav_frame = ttk.Frame(right_panel, style='Card.TFrame')
        nav_frame.pack(fill='x', pady=(10, 0))

        # Use grid for proper 3-column layout
        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)
        nav_frame.columnconfigure(2, weight=1)

        self.prev_btn = ttk.Button(
            nav_frame,
            text="← Previous",
            command=self._prev_image,
            style='TButton'
        )
        self.prev_btn.grid(row=0, column=0, sticky='w')

        self.image_counter_label = ttk.Label(
            nav_frame,
            text="",
            style='Card.TLabel'
        )
        self.image_counter_label.grid(row=0, column=1)

        self.next_btn = ttk.Button(
            nav_frame,
            text="Next →",
            command=self._next_image,
            style='TButton'
        )
        self.next_btn.grid(row=0, column=2, sticky='e')

        # Bind keyboard arrows for navigation
        self.root.bind('<Left>', lambda e: self._prev_image())
        self.root.bind('<Right>', lambda e: self._next_image())

    def _on_preview_resize(self, event):
        """Handle canvas resize - reposition placeholder or resize current image."""
        # Center the placeholder text
        self.preview_canvas.coords('placeholder', event.width // 2, event.height // 2)

        # If there's a current image, resize it
        if self.current_preview_image:
            self._display_current_image()

    def _display_preview_image(self, image_path: str, filename: str):
        """Display an image in the preview panel and add to navigation list."""
        try:
            # Add to navigation list
            self.debug_image_list.append((image_path, filename))
            self.current_image_index = len(self.debug_image_list) - 1

            # Load and store the original image path for resizing
            self._current_image_path = image_path
            self._current_filename = filename

            # Update filename label
            self.preview_filename_label.config(text=filename)

            # Update counter
            self._update_image_counter()

            # Remove placeholder
            self.preview_canvas.delete('placeholder')

            self._display_current_image()

        except Exception as e:
            print(f"Error displaying preview: {e}")

    def _update_image_counter(self):
        """Update the image counter label."""
        if self.debug_image_list:
            self.image_counter_label.config(
                text=f"{self.current_image_index + 1} / {len(self.debug_image_list)}"
            )
        else:
            self.image_counter_label.config(text="")

    def _prev_image(self):
        """Navigate to previous image."""
        if not self.debug_image_list or self.current_image_index <= 0:
            return

        self.current_image_index -= 1
        path, filename = self.debug_image_list[self.current_image_index]
        self._current_image_path = path
        self._current_filename = filename
        self.preview_filename_label.config(text=filename)
        self._update_image_counter()
        self._display_current_image()

    def _next_image(self):
        """Navigate to next image."""
        if not self.debug_image_list or self.current_image_index >= len(self.debug_image_list) - 1:
            return

        self.current_image_index += 1
        path, filename = self.debug_image_list[self.current_image_index]
        self._current_image_path = path
        self._current_filename = filename
        self.preview_filename_label.config(text=filename)
        self._update_image_counter()
        self._display_current_image()

    def _display_current_image(self):
        """Display/resize the current image to fit the canvas."""
        if not hasattr(self, '_current_image_path'):
            return

        try:
            # Get canvas dimensions
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()

            if canvas_width < 10 or canvas_height < 10:
                return

            # Load image
            img = Image.open(self._current_image_path)

            # Calculate scale to fit while maintaining aspect ratio
            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height)

            # Don't upscale small images
            scale = min(scale, 1.0)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            self.current_preview_image = ImageTk.PhotoImage(img)

            # Clear canvas and display centered image
            self.preview_canvas.delete('all')
            x = canvas_width // 2
            y = canvas_height // 2
            self.preview_canvas.create_image(x, y, image=self.current_preview_image, anchor='center')

        except Exception as e:
            print(f"Error resizing preview: {e}")

    def _browse_folder(self):
        """Open folder browser dialog."""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.folder_path.set(folder)
            self._update_image_count()

    def _update_image_count(self):
        """Update the status with current image count based on settings."""
        folder = self.folder_path.get()
        if folder:
            recursive = self.include_subfolders.get()
            images = get_image_files(folder, recursive=recursive)
            suffix = " (including subfolders)" if recursive else ""
            self._set_status(f"Found {len(images)} images{suffix}")

    def _log(self, message: str):
        """Append message to log."""
        self.log_text.config(state='normal')
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')
        self.log_text.config(state='disabled')
        self.root.update_idletasks()

    def _set_status(self, message: str):
        """Update status bar."""
        self.status_label.config(text=message)

    def _set_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = current
        percent = int((current / total) * 100) if total > 0 else 0
        self.progress_label.config(text=f"Processing: {current}/{total} ({percent}%)")

    def _cancel_processing(self):
        """Request cancellation of processing."""
        self.cancel_requested = True
        self._log("\n*** Cancellation requested, stopping after current image... ***")
        self.cancel_btn.config(state='disabled')

    def _start_processing(self):
        """Start the processing in a background thread."""
        folder = self.folder_path.get()

        if not folder:
            messagebox.showerror("Error", "Please select an image folder")
            return

        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Selected folder does not exist")
            return

        recursive = self.include_subfolders.get()
        images = get_image_files(folder, recursive=recursive)
        if not images:
            messagebox.showerror("Error", "No images found in folder")
            return

        # Disable UI during processing
        self.processing = True
        self.cancel_requested = False
        self.start_btn.config(state='disabled')
        self.folder_entry.config(state='disabled')
        self.csv_check.config(state='disabled')
        self.preview_check.config(state='disabled')
        self.save_debug_check.config(state='disabled')
        self.metadata_check.config(state='disabled')
        self.subfolders_check.config(state='disabled')

        # Show cancel button, hide open folder button
        self.open_folder_btn.pack_forget()
        self.cancel_btn.pack(side='left', padx=(10, 0))
        self.cancel_btn.config(state='normal')

        # Clear log and image list
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.config(state='disabled')
        self.debug_image_list = []
        self.current_image_index = 0
        self._update_image_counter()

        # Start processing thread
        thread = threading.Thread(
            target=self._process_images,
            args=(folder, images),
            daemon=True
        )
        thread.start()

    def _process_images(self, folder: str, images: list[Path]):
        """Process images in background thread."""
        root_folder = Path(folder)

        try:
            # Initialize tagger
            self._log("Initializing bib detector...")
            model_path = get_model_path()

            def log_callback(msg):
                self.root.after(0, lambda: self._log(msg))

            self.tagger = BibTagger(
                model_path=model_path,
                progress_callback=log_callback
            )

            # Create debug folder if saving debug images to disk
            debug_folder = None
            if self.save_debug_images.get():
                debug_folder = root_folder / 'debug_images'
                debug_folder.mkdir(exist_ok=True)
                self._log(f"Debug images will be saved to: debug_images/")

            # Process images
            results = []
            total = len(images)
            successful = 0
            total_bibs = 0
            images_with_bibs = 0

            self.root.after(0, lambda: self._log(f"\nProcessing {total} image(s)...\n"))

            cancelled = False
            for i, img_path in enumerate(images, 1):
                # Check for cancellation
                if self.cancel_requested:
                    cancelled = True
                    self.root.after(0, lambda: self._log("\n*** Processing cancelled ***"))
                    break

                # Calculate relative path from root folder
                try:
                    rel_path = img_path.relative_to(root_folder)
                except ValueError:
                    rel_path = img_path.name

                # Update progress bar and status
                self.root.after(0, lambda c=i, t=total, tb=total_bibs: (
                    self._set_progress(c, t),
                    self._set_status(f"Processing {c}/{t} | {tb} bibs found so far")
                ))

                # Log current file
                self.root.after(0, lambda p=rel_path, idx=i, t=total: self._log(f"Processing {idx}/{t}: {p}"))

                # Process (pass write_metadata option)
                result = self.tagger.process_image(str(img_path), write_metadata=self.write_metadata.get())
                result['relative_path'] = str(rel_path)
                results.append(result)

                # Build detailed status line
                if result['bibs']:
                    bib_count = len(result['bibs'])
                    bibs_str = ', '.join(b['number'] for b in result['bibs'])
                    line = f"  → Bibs detected: {bib_count} | Numbers: {bibs_str}"
                    total_bibs += bib_count
                    images_with_bibs += 1
                else:
                    line = f"  → No bibs detected"

                if result['error']:
                    line += f" | Error: {result['error']}"

                self.root.after(0, lambda l=line: self._log(l))

                if result['success']:
                    successful += 1

                # Handle debug images: save to disk and/or show preview
                show_preview = self.show_preview.get()
                save_to_disk = self.save_debug_images.get()

                if save_to_disk or show_preview:
                    if save_to_disk and debug_folder:
                        # Create subdirectory structure in debug folder
                        if isinstance(rel_path, Path) and rel_path.parent != Path('.'):
                            debug_subdir = debug_folder / rel_path.parent
                            debug_subdir.mkdir(parents=True, exist_ok=True)
                        else:
                            debug_subdir = debug_folder

                        debug_path = debug_subdir / f"{img_path.stem}_debug{img_path.suffix}"
                        self.tagger.save_debug_image(result, str(debug_path))

                        # Show saved image in preview if enabled
                        if show_preview:
                            self.root.after(0, lambda p=str(debug_path), f=str(rel_path): self._display_preview_image(p, f))

                    elif show_preview:
                        # Generate temp debug image for preview only (not saving to disk)
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=img_path.suffix, delete=False) as tmp:
                            temp_debug_path = tmp.name
                        self.tagger.save_debug_image(result, temp_debug_path)
                        self.root.after(0, lambda p=temp_debug_path, f=str(rel_path): self._display_preview_image(p, f))

            # How many were actually processed
            processed_count = len(results)

            # Generate CSV if enabled (with relative paths) - even if cancelled, save what we have
            if self.generate_csv.get() and results:
                csv_path = root_folder / 'bib_results.csv'
                self._write_csv(results, str(csv_path), use_relative_paths=True)

            # Summary
            status_word = "Cancelled" if cancelled else "Complete"
            summary = f"""
{'='*50}
  Processing {status_word}
{'='*50}
  Images processed:    {processed_count}/{total}
  Images with bibs:    {images_with_bibs}
  Total bibs found:    {total_bibs}
{'='*50}"""
            self.root.after(0, lambda: self._log(summary))
            self.root.after(0, lambda: self._set_status(f"{status_word}! Processed {processed_count}/{total} images, found {total_bibs} bibs"))
            self.root.after(0, lambda: self.progress_label.config(text=f"{status_word}: {processed_count}/{total} images processed"))

        except Exception as e:
            self.root.after(0, lambda: self._log(f"\nError: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            # Re-enable UI
            self.root.after(0, self._reset_ui)

    def _write_csv(self, results: list[dict], csv_path: str, use_relative_paths: bool = False):
        """Write results to CSV file."""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Path' if use_relative_paths else 'Filename',
                'Bib Numbers',
                'Detection Count',
                'Success',
                'Time (ms)',
                'Error'
            ])

            for r in results:
                bibs = ', '.join(b['number'] for b in r['bibs']) if r['bibs'] else ''
                # Use relative path if available, otherwise filename
                path_col = r.get('relative_path', r['filename']) if use_relative_paths else r['filename']
                writer.writerow([
                    path_col,
                    bibs,
                    r['detections'],
                    r['success'],
                    f"{r['time_ms']:.1f}",
                    r['error'] or ''
                ])

    def _reset_ui(self):
        """Reset UI after processing completes."""
        self.processing = False
        self.cancel_requested = False
        self.start_btn.config(state='normal')
        self.folder_entry.config(state='normal')
        self.csv_check.config(state='normal')
        self.preview_check.config(state='normal')
        self.save_debug_check.config(state='normal')
        self.metadata_check.config(state='normal')
        self.subfolders_check.config(state='normal')
        # Hide cancel button, show Open Folder button
        self.cancel_btn.pack_forget()
        self.open_folder_btn.pack(side='left', padx=(10, 0))

    def _open_folder(self):
        """Open the processed folder in the system file manager."""
        folder = self.folder_path.get()
        if not folder or not os.path.isdir(folder):
            return

        if sys.platform == 'darwin':
            subprocess.run(['open', folder])
        elif sys.platform == 'win32':
            subprocess.run(['explorer', folder])
        else:
            # Linux - try xdg-open
            subprocess.run(['xdg-open', folder])

    def run(self):
        """Start the main event loop."""
        # Set dark title bar on Windows 11
        if sys.platform == 'win32':
            try:
                import ctypes
                self.root.update()
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                    ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int)
                )
            except:
                pass

        self.root.mainloop()


def main():
    app = BibTaggerApp()
    app.run()


if __name__ == '__main__':
    main()
