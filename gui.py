import numpy as np
import tkinter as tk
import tensorflow as tf
from PIL import Image, ImageTk
from tkinter import filedialog
from tensorflow.keras.applications.efficientnet import preprocess_input

# LOAD CNN MODEL
model = tf.keras.models.load_model("effNet.keras") #THE GUI ONLY USES THE MODEL FOR E2E EFFNET; TO USE THE E2E RESNET UNCOMMENT THE LAST LINE OF CODE IN ITS CELL
# CLASS NAMES
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]
# GUI WINDOW
root = tk.Tk()
root.title("Brain MRI Detection")
root.geometry("700x700")
# TITLE
title_label = tk.Label(root, text="Brain MRI Tumor Detection", font=("Arial", 20, "bold"))
title_label.pack(pady=20)
# IMAGE DISPLAY
image_label = tk.Label(root)
image_label.pack(pady=10)
# PREDICTION LABEL
prediction_label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 16)
)
prediction_label.pack(pady=10)
# PROBABILITY LABEL
probability_label = tk.Label(
    root,
    text="",
    font=("Arial", 13),
    justify="left"
)
probability_label.pack(pady=10)
# IMAGE UPLOAD FUNCTION
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )
    if not file_path:
        return
    # Load image
    image = Image.open(file_path).convert("RGB")
    # Show image in GUI
    display_image = image.resize((300, 300))
    photo = ImageTk.PhotoImage(display_image)
    image_label.config(image=photo)
    image_label.image = photo
    # PREPROCESS IMAGE
    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # PREDICTION
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    # SHOW RESULTS
    prediction_label.config(
        text=f"Prediction: {class_names[predicted_class]}"
    )
    probability_text = ""
    for i, prob in enumerate(predictions):
        probability_text += (
            f"{class_names[i]}: "
            f"{prob:.3f}\n"
        )
    probability_label.config(text=probability_text)

# UPLOAD BUTTON
upload_button = tk.Button(root, text="Upload MRI Image", command=upload_image, font=("Arial", 14))
upload_button.pack(pady=20)
# START GUI
root.mainloop()