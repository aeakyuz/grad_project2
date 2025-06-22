import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf

from apply_filter import apply_filter_tiled, preprocess_image, postprocess_output
from filter_library import FILTER_FUNCTIONS

class FilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter Applicator & Evaluator")
        self.root.geometry("1600x900")

        self.input_path = tk.StringVar()
        self.models_dir_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.use_tiled_inference = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Ready")
        self.ssim_score_text = tk.StringVar(value="SSIM: -")
        self.selected_filter = tk.StringVar()

        self.model_map = {
            "gaussian": "blur.keras",
            "cartoon": "cartoon.keras",
            "grayscale": "grayscale.h5",
            "oil": "oil.h5",
            "sepia": "sepia.h5",
            "sketch": "sketch.h5"
        }

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(side="left", fill="y")
        image_frame = ttk.Frame(main_frame, padding="10")
        image_frame.pack(side="right", fill="both", expand=True)

        ttk.Label(control_frame, text="1. Select Input Image:").pack(anchor="w", pady=2)
        ttk.Entry(control_frame, textvariable=self.input_path, width=50).pack(anchor="w")
        ttk.Button(control_frame, text="Browse...", command=self.select_input_image).pack(anchor="w", pady=(0, 10))

        ttk.Label(control_frame, text="2. Select Models Directory:").pack(anchor="w", pady=2)
        ttk.Entry(control_frame, textvariable=self.models_dir_path, width=50).pack(anchor="w")
        ttk.Button(control_frame, text="Browse...", command=self.select_models_dir).pack(anchor="w", pady=(0, 10))

        ttk.Label(control_frame, text="3. Select Filter Type:").pack(anchor="w", pady=2)
        self.filter_combobox = ttk.Combobox(control_frame, textvariable=self.selected_filter, width=48, state="readonly")
        self.filter_combobox['values'] = list(FILTER_FUNCTIONS.keys())
        self.filter_combobox.pack(anchor="w", pady=(0, 10))
        self.filter_combobox.bind('<<ComboboxSelected>>', self.on_filter_selected)

        ttk.Label(control_frame, text="4. Trained Model File (Auto-selected):").pack(anchor="w", pady=2)
        ttk.Entry(control_frame, textvariable=self.model_path, width=50).pack(anchor="w")
        ttk.Button(control_frame, text="Browse...", command=self.select_model).pack(anchor="w", pady=(0, 10))
        
        ttk.Label(control_frame, text="5. Select Output Location:").pack(anchor="w", pady=2)
        ttk.Entry(control_frame, textvariable=self.output_path, width=50).pack(anchor="w")
        ttk.Button(control_frame, text="Browse...", command=self.select_output_path).pack(anchor="w", pady=(0, 20))

        ttk.Checkbutton(control_frame, text="Use Tiled Inference (High Quality)", variable=self.use_tiled_inference).pack(anchor="w", pady=10)
        ttk.Button(control_frame, text="Apply & Compare", command=self.start_inference_thread, style="Accent.TButton").pack(anchor="w", pady=20, ipady=5)

        status_frame = ttk.Frame(self.root, relief="sunken")
        status_frame.pack(side="bottom", fill="x")
        status_bar = ttk.Label(status_frame, textvariable=self.status_text, anchor="w", padding="5")
        status_bar.pack(side="left", fill="x", expand=True)
        ssim_label = ttk.Label(status_frame, textvariable=self.ssim_score_text, anchor="e", padding="5")
        ssim_label.pack(side="right")
        self.input_image_label = ttk.Label(image_frame, text="Input Image")
        self.input_image_label.pack(side="left", fill="both", expand=True, padx=5)
        self.ground_truth_label = ttk.Label(image_frame, text="Ground Truth (Real Filter)")
        self.ground_truth_label.pack(side="left", fill="both", expand=True, padx=5)
        self.output_image_label = ttk.Label(image_frame, text="Inferred Output (Model)")
        self.output_image_label.pack(side="right", fill="both", expand=True, padx=5)
        self.model = None
        self.last_model_path = ""

    def select_input_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            self.input_path.set(path)
            self.display_image(self.input_image_label, path=path)
            self.ssim_score_text.set("SSIM: -")
            self.ground_truth_label.config(image='', text="Ground Truth (Real Filter)")
            self.output_image_label.config(image='', text="Inferred Output (Model)")

    def select_models_dir(self):
        path = filedialog.askdirectory(title="Select Folder Containing Your Models")
        if path:
            self.models_dir_path.set(path)
            if self.selected_filter.get():
                self.on_filter_selected(None)

    def on_filter_selected(self, event):
        filter_key = self.selected_filter.get()
        models_dir = self.models_dir_path.get()
        
        if not models_dir:
            messagebox.showwarning("Warning", "Please select the models directory first.")
            self.selected_filter.set("")
            return

        if filter_key in self.model_map:
            model_filename = self.model_map[filter_key]
            full_path = os.path.join(models_dir, model_filename)
            
            if os.path.exists(full_path):
                self.model_path.set(full_path)
            else:
                self.model_path.set("")
                messagebox.showwarning("Model Not Found", f"Could not find '{model_filename}' in the selected directory.")
        else:
            self.model_path.set("")

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras *.h5")])
        if path:
            self.model_path.set(path)

    def select_output_path(self):
        input_filename = os.path.basename(self.input_path.get())
        suggested_filename = "output.png"
        if '.' in input_filename:
            base, ext = input_filename.rsplit('.', 1)
            suggested_filename = f"{base}_filtered.{ext}"
            
        path = filedialog.asksaveasfilename(initialfile=suggested_filename, defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if path:
            self.output_path.set(path)

    def display_image(self, label_widget, img_data=None, path=None, max_size=(500, 500)):
        try:
            if path:
                img_pil = Image.open(path)
            elif img_data is not None:
                if len(img_data.shape) == 2 or img_data.shape[2] == 1:
                    img_pil = Image.fromarray(img_data)
                else:
                    img_pil = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
            else:
                return

            img_pil.thumbnail(max_size)
            photo = ImageTk.PhotoImage(img_pil)
            label_widget.config(image=photo, text="")
            label_widget.image = photo
        except Exception as e:
            messagebox.showerror("Display Error", f"Could not display image: {e}")

    def calculate_ssim(self, ground_truth_img, generated_img):
        try:
            common_dim = (512, 512)
            gt_resized = cv2.resize(ground_truth_img, common_dim)
            gen_resized = cv2.resize(generated_img, common_dim)

            gt_shape = gt_resized.shape
            gen_shape = gen_resized.shape

            is_gt_color = len(gt_shape) == 3 and gt_shape[2] == 3
            is_gen_color = len(gen_shape) == 3 and gen_shape[2] == 3

            if is_gt_color and not is_gen_color:
                gt_resized = cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY)
            elif not is_gt_color and is_gen_color:
                gen_resized = cv2.cvtColor(gen_resized, cv2.COLOR_BGR2GRAY)
            
            is_multichannel = (len(gt_resized.shape) == 3 and gt_resized.shape[2] == 3)
            
            if not is_multichannel and (is_gt_color or is_gen_color):
                 score, _ = ssim(gt_resized, gen_resized, full=True)
            else:
                score, _ = ssim(gt_resized, gen_resized, full=True, channel_axis=2 if is_multichannel else None)

            self.ssim_score_text.set(f"SSIM: {score:.4f}")

        except Exception as e:
            self.ssim_score_text.set("SSIM: Error")
            print(f"Could not calculate SSIM: {e}")

    def start_inference_thread(self):
        self.status_text.set("Processing...")
        self.ssim_score_text.set("SSIM: -")
        threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        in_path = self.input_path.get()
        out_path = self.output_path.get()
        mod_path = self.model_path.get()
        filter_key = self.selected_filter.get()

        if not all([in_path, out_path, mod_path, filter_key]):
            messagebox.showerror("Error", "Please select an input image, filter type, model, and output path.")
            self.status_text.set("Ready")
            return

        try:
            self.status_text.set(f"Applying real '{filter_key}' filter...")
            original_image_cv = cv2.imread(in_path)
            if original_image_cv is None: raise Exception(f"Failed to load input image: {in_path}")
            
            filter_function = FILTER_FUNCTIONS[filter_key]
            ground_truth_image = filter_function(original_image_cv)
            self.display_image(self.ground_truth_label, img_data=ground_truth_image)

            self.status_text.set("Loading model and running inference...")
            if self.model is None or self.last_model_path != mod_path:
                self.model = tf.keras.models.load_model(mod_path, compile=False)
                self.last_model_path = mod_path

            model_height = self.model.input_shape[1]
            if self.use_tiled_inference.get():
                inferred_image_rgb = apply_filter_tiled(self.model, in_path, patch_size=model_height)
            else:
                input_tensor, orig_h, orig_w = preprocess_image(in_path, model_height, model_height)
                prediction = self.model.predict(input_tensor)
                inferred_image_rgb = postprocess_output(prediction, orig_h, orig_w)

            if inferred_image_rgb is None: raise Exception("Inference failed.")
            
            if len(inferred_image_rgb.shape) == 3 and inferred_image_rgb.shape[2] == 3:
                inferred_image_bgr = cv2.cvtColor(inferred_image_rgb, cv2.COLOR_RGB2BGR)
            else:
                inferred_image_bgr = inferred_image_rgb

            self.display_image(self.output_image_label, img_data=inferred_image_bgr)
            cv2.imwrite(out_path, inferred_image_bgr)
            
            self.status_text.set("Success! Calculating SSIM...")
            self.calculate_ssim(ground_truth_image, inferred_image_bgr)
            self.status_text.set(f"Success! Image saved to {out_path}")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            self.status_text.set("Error during processing. Ready.")


if __name__ == "__main__":
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")
    except ImportError:
        root = tk.Tk()
        style = ttk.Style(root)
        style.configure("Accent.TButton", foreground="white", background="blue")

    app = FilterApp(root)
    root.mainloop()

