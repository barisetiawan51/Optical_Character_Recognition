import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageTk
from tkinter import (
    Tk,
    Label,
    Button,
    filedialog,
    Canvas,
    Scrollbar,
    Radiobutton,
    Frame,
    StringVar,
    font,
)

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Imam Bari Setiawan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Initialize the zoom scale
zoom_scale = 1.0

# Store the original image
original_image = None

# Store the selected blur type
selected_blur_type = None
selected_dilation = False
selected_inversion = False
selected_threshold_type = None


def open_image():
    # Open file dialog to select an image
    filename = filedialog.askopenfilename(
        initialdir="/",
        title="Select Image",
        filetypes=(("Image Files", ".png *.jpg *.jpeg *.bmp"),
                   ("All Files", ".*")),
    )
    # Read the image using OpenCV
    image = cv2.imread(filename)

    # Store the original image
    global original_image
    original_image = image.copy()

    # Display the image in the GUI with the desired size
    display_image(image, (600, 400))


def display_image(image, size):
    # Resize image based on the desired size
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    # Convert the resized image to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    # Create a PhotoImage object to display in the label
    image_tk = ImageTk.PhotoImage(image_pil)

    # Update the image label with the new image
    canvas.create_image(0, 0, anchor="nw", image=image_tk)
    canvas.config(scrollregion=canvas.bbox("all"))

    # Configure scroll region to include the image
    canvas.config(scrollregion=canvas.bbox("all"))

    # Update the image label with the new image
    canvas.image = image_tk  # type: ignore


def sharpen_image():
    global original_image
    if original_image is not None:
        # Create a sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        # Apply the sharpening filter to the original image
        sharpened_image = cv2.filter2D(original_image, -1, kernel)

        # Display the sharpened image
        display_image(sharpened_image, (600, 400))


def apply_blur():
    global selected_blur_type, original_image, selected_dilation
    if original_image is not None and selected_blur_type is not None:
        # Apply the selected blur type to the original image
        # Convert the image to grayscale
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        if selected_dilation:
            # Apply dilation to the original image
            # Apply dilation to the grayscale image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated_image = cv2.dilate(gray, kernel, iterations=1)
            # Convert the dilated image back to BGR color format
            dilated_image_bgr = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)
            blurred_image = dilated_image_bgr  # Assign dilated image to blurred_image
            # Display the dilated image
            display_image(dilated_image_bgr, (600, 400))
        else:
            # Apply the selected blur type to the original image
            if selected_blur_type == "Gaussian":
                blurred_image = cv2.GaussianBlur(original_image, (11, 11), 0)
            elif selected_blur_type == "Median":
                blurred_image = cv2.medianBlur(original_image, 11)
            elif selected_blur_type == "Bilateral":
                blurred_image = cv2.bilateralFilter(original_image, 9, 75, 75)
            else:
                blurred_image = original_image.copy()

            # Display the blurred image
            display_image(blurred_image, (600, 400))

        # Apply inversion if selected
        if selected_inversion:
            inverted_image = cv2.bitwise_not(blurred_image)
            blurred_image = inverted_image  # Assign inverted image to blurred_image
            # Display the inverted image
            display_image(inverted_image, (600, 400))


def select_blur_type(blur_type):
    global selected_blur_type
    selected_blur_type = blur_type


def select_dilation():
    global selected_dilation
    selected_dilation = True


def select_non_dilation():
    global selected_dilation
    selected_dilation = False


def select_inversion():
    global selected_inversion
    selected_inversion = True


def select_non_inversion():
    global selected_inversion
    selected_inversion = False


def apply_threshold():
    global selected_threshold_type, original_image
    if original_image is not None and selected_threshold_type is not None:
        # Apply the selected threshold type to the original image
        # Convert the image to grayscale
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Apply the selected threshold type
        if selected_threshold_type == "Binary":
            _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        elif selected_threshold_type == "Binary Inverted":
            _, thresholded = cv2.threshold(
                gray, 128, 255, cv2.THRESH_BINARY_INV)
        elif selected_threshold_type == "Truncate":
            _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_TRUNC)
        elif selected_threshold_type == "To Zero":
            _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_TOZERO)
        elif selected_threshold_type == "To Zero Inverted":
            _, thresholded = cv2.threshold(
                gray, 128, 255, cv2.THRESH_TOZERO_INV)
        else:
            thresholded = gray

        # Display the processed image
        display_image(thresholded, (600, 400))


def select_threshold_type(threshold_type):
    global selected_threshold_type
    selected_threshold_type = threshold_type


def reset_image():
    global original_image, selected_blur_type, selected_dilation, selected_inversion
    if original_image is not None:
        selected_blur_type = None
        selected_dilation = False
        selected_inversion = False
        display_image(original_image, (600, 400))


def ocr():
    global original_image
    if original_image is not None:
        # Convert the image to grayscale
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(Image.fromarray(gray))
        # Display the OCR result in the GUI
        result_label.config(text=text)


def zoom_in():
    global zoom_scale, original_image, select_blur_type, select_threshold_type
    zoom_scale += 0.1
    # Calculate the new size based on the zoom scale
    new_size = (int(600 * zoom_scale), int(400 * zoom_scale))
    # Reload the original image to apply the new size
    if original_image is not None:
        display_image(original_image, new_size)
    # Reload the blurred image to apply the new size
    if select_blur_type is not None:
        display_image(select_blur_type, new_size)
    # Reload the thresholded image to apply the new size
    if select_threshold_type is not None:
        display_image(select_threshold_type, new_size)


def zoom_out():
    global zoom_scale, original_image, select_blur_type, select_threshold_type
    zoom_scale = max(zoom_scale - 0.1, 0.1)
    # Calculate the new size based on the zoom scale
    new_size = (int(600 * zoom_scale), int(400 * zoom_scale))
    # Reload the original image to apply the new size
    if original_image is not None:
        display_image(original_image, new_size)
    # Reload the blurred image to apply the new size
    if select_blur_type is not None:
        display_image(select_blur_type, new_size)
    # Reload the thresholded image to apply the new size
    if select_threshold_type is not None:
        display_image(select_threshold_type, new_size)


# Create the main window
root = Tk()
root.title("OCR and Image Processing")
root.geometry("800x600")

# Create a frame to hold the OCR result and image label
result_frame = Frame(root)
result_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Create a font with the desired size
label_font = font.Font(size=8)

# Create a label to display the OCR result
result_label = Label(result_frame, text="", wraplength=1000,
                     font=label_font, justify="left")
result_label.pack(side="right", padx=(0, 10))

# Create a canvas to display the image with scrollbars
canvas = Canvas(result_frame, width=600, height=200)
canvas.pack(side="left", fill="both", expand=True)

# Create a horizontal scrollbar for the canvas
horizontal_scrollbar = Scrollbar(
    result_frame, orient="horizontal", command=canvas.xview
)
horizontal_scrollbar.pack(side="bottom", fill="x")

# Configure the canvas to work with the scrollbar
canvas.config(xscrollcommand=horizontal_scrollbar.set)
canvas.config(scrollregion=canvas.bbox("all"))

# Create a vertical scrollbar for the canvas
vertical_scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
vertical_scrollbar.pack(side="right", fill="y")

# Configure the canvas to work with the scrollbar
canvas.config(yscrollcommand=vertical_scrollbar.set)

# Create a button to open an image
open_button = Button(root, text="Open Image", command=open_image)
open_button.pack(side="top", pady=(5, 5), padx=(10, 10), anchor="center")

# Create threshold buttons
threshold_frame = Frame(root)
threshold_frame.pack()

# Create radio buttons for threshold types
threshold_var = StringVar()

threshold_label = Label(threshold_frame, text="Threshold citra:")
threshold_label.pack(padx=(0, 10))

threshold_radio1 = Radiobutton(
    threshold_frame,
    text="Binary",
    variable=threshold_var,
    value="Binary",
    command=lambda: select_threshold_type(threshold_var.get()),
)
threshold_radio1.pack(side="right", padx=(0, 10))

threshold_radio2 = Radiobutton(
    threshold_frame,
    text="Binary Inverted",
    variable=threshold_var,
    value="Binary Inverted",
    command=lambda: select_threshold_type(threshold_var.get()),
)
threshold_radio2.pack(side="right", padx=(0, 5))

threshold_radio3 = Radiobutton(
    threshold_frame,
    text="Truncate",
    variable=threshold_var,
    value="Truncate",
    command=lambda: select_threshold_type(threshold_var.get()),
)
threshold_radio3.pack(side="left", padx=(0, 5))

threshold_radio4 = Radiobutton(
    threshold_frame,
    text="To Zero",
    variable=threshold_var,
    value="To Zero",
    command=lambda: select_threshold_type(threshold_var.get()),
)
threshold_radio4.pack(side="left", padx=(0, 5))

threshold_radio5 = Radiobutton(
    threshold_frame,
    text="To Zero Inverted",
    variable=threshold_var,
    value="To Zero Inverted",
    command=lambda: select_threshold_type(threshold_var.get()),
)
threshold_radio5.pack(side="left", padx=(0, 5))

# Create a label for the "Efek" text
efek_label = Label(root, text="segmentasi citra:")
efek_label.pack(padx=(0, 10))

# Create a frame to hold blur type selection
blur_type_frame = Frame(root)
blur_type_frame.pack()

# Create radio buttons for blur types
blur_type_var = StringVar()

gaussian_button = Radiobutton(
    blur_type_frame,
    text="Gaussian",
    variable=blur_type_var,
    value="Gaussian",
    command=lambda: select_blur_type(blur_type_var.get()),
)
gaussian_button.pack(side="left", padx=(0, 10))

median_button = Radiobutton(
    blur_type_frame,
    text="Median",
    variable=blur_type_var,
    value="Median",
    command=lambda: select_blur_type(blur_type_var.get()),
)
median_button.pack(side="left", padx=(0, 10))

bilateral_button = Radiobutton(
    blur_type_frame,
    text="Bilateral",
    variable=blur_type_var,
    value="Bilateral",
    command=lambda: select_blur_type(blur_type_var.get()),
)
bilateral_button.pack(side="left", padx=(0, 10))

# Create radio buttons for dilation
dilation_var = StringVar()

dilation_button = Radiobutton(
    blur_type_frame,
    text="Dilation",
    variable=dilation_var,
    value="Dilation",
    command=select_dilation,
)
dilation_button.pack(side="left", padx=(0, 10))

non_dilation_button = Radiobutton(
    blur_type_frame,
    text="Non-Dilation",
    variable=dilation_var,
    value="Non-Dilation",
    command=select_non_dilation,
)
non_dilation_button.pack(side="left", padx=(0, 10))

# Create a frame to hold inversion selection
inversion_frame = Frame(root)
inversion_frame.pack()

# Create radio buttons for inversion
inversion_var = StringVar()

inversion_button = Radiobutton(
    inversion_frame,
    text="Inversion",
    variable=inversion_var,
    value="Inversion",
    command=select_inversion,
)
inversion_button.pack(side="left", padx=(0, 10))

non_inversion_button = Radiobutton(
    inversion_frame,
    text="Non-Inversion",
    variable=inversion_var,
    value="Non-Inversion",
    command=select_non_inversion,
)
non_inversion_button.pack(side="left", padx=(0, 10))

# Create buttons for zooming
zoom_frame = Frame(root)
zoom_frame.pack(side="right", pady=(0, 5), padx=(0, 10))

zoom_in_button = Button(zoom_frame, text="Zoom In", command=zoom_in)
zoom_in_button.pack(side="left", padx=(0, 10))

zoom_out_button = Button(zoom_frame, text="Zoom Out", command=zoom_out)
zoom_out_button.pack(side="left")

# Create a button to reset the image
reset_button = Button(root, text="Reset Image", command=reset_image)
reset_button.pack(side="right", pady=(31, 35), padx=(0, 10))

# Create a button to apply blur
apply_blur_button = Button(root, text="Apply Blur", command=apply_blur)
apply_blur_button.pack(side="right", pady=(31, 35), padx=(0, 10))

threshold_button = Button(
    root, text="Apply Thresholding", command=apply_threshold)
threshold_button.pack(side="right", pady=(31, 35), padx=(0, 10))

# Create a button to sharpen the image
sharpen_button = Button(root, text="Apply Sharpen", command=sharpen_image)
sharpen_button.pack(side="right", pady=(31, 35), padx=(0, 10))

# Create a button to perform OCR
ocr_button = Button(root, text="Image to Text", command=ocr)
ocr_button.pack(side="right", pady=(31, 35), padx=(0, 10))

# Start the GUI event loop
root.mainloop()
