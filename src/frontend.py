import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Car Detection App",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Styling (using st.markdown for custom CSS) ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1a202c;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    h2 {
        color: #2d3748;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stFileUploader label {
        color: #2d3748;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stAlert {
        border-radius: 8px;
    }
    .css-1r6dn7c { /* Targeting the image container */
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- App Title and Description ---
st.title("🚗 Car Detection App")
st.write("Upload an image to detect if it contains a car and its bounding box.")
st.write("---")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image file (JPG, JPEG, or PNG) for car detection."
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("") # Add some space

    if st.button("Detect Car"):
        with st.spinner("Detecting car..."):
            try:
                # Prepare the file for sending to FastAPI
                files = {"file": uploaded_file.getvalue()}

                # Make the POST request to your FastAPI endpoint
                # Make sure your FastAPI app is running on this address
                FASTAPI_URL = "http://127.0.0.1:8000/predict/"
                response = requests.post(FASTAPI_URL, files=files)

                # Check if the request was successful
                if response.status_code == 200:
                    result = response.json()
                    st.success("Detection Complete!")

                    if result.get("Detect") == "Car detected":
                        st.subheader("🎉 Car Detected!")
                        bbox_data = result.get("bounding box")
                        if bbox_data:
                            st.write("Bounding Box Coordinates (denormalized):")
                            st.json(bbox_data)

                            # Draw bounding box on the image
                            original_image = Image.open(uploaded_file)
                            draw = ImageDraw.Draw(original_image)

                            # Extract denormalized coordinates
                            xmin = bbox_data["xmin"]
                            ymin = bbox_data["ymin"]
                            xmax = bbox_data["xmax"]
                            ymax = bbox_data["ymax"]

                            # Define box color and width
                            box_color = "red"
                            box_width = 3

                            # Draw the rectangle
                            # The bbox coordinates are expected to be pixel values relative to the 299x299 input image.
                            # We need to scale them to the original image's dimensions.
                            original_width, original_height = original_image.size

                            # Calculate scaling factors
                            # Your FastAPI preprocess_layer resizes to 299x299
                            scale_x = original_width / 299.0
                            scale_y = original_height / 299.0

                            # Apply scaling to the bbox coordinates
                            # Assuming the bbox coordinates from FastAPI are for a 299x299 image
                            # and need to be scaled back to original image dimensions for drawing.
                            # If FastAPI returns coordinates relative to the original image,
                            # then scaling is not needed here.
                            # Based on your denormalize_bbox, it multiplies by 299, so it returns
                            # coordinates for a 299x299 image.
                            # Let's scale these 299x299 coords to the original image size.
                            scaled_xmin = xmin * scale_x
                            scaled_ymin = ymin * scale_y
                            scaled_xmax = xmax * scale_x
                            scaled_ymax = ymax * scale_y


                            draw.rectangle(
                                [(scaled_xmin, scaled_ymin), (scaled_xmax, scaled_ymax)],
                                outline=box_color,
                                width=box_width
                            )

                            st.subheader("Image with Bounding Box:")
                            st.image(original_image, caption="Detected Car with Bounding Box", use_column_width=True)

                    else:
                        st.subheader("😞 No Car Detected.")

                else:
                    st.error(f"Error: Could not connect to FastAPI. Status code: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    st.warning("Please ensure your FastAPI application is running at `http://127.0.0.1:8000`.")

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the FastAPI server.")
                st.warning("Please ensure your FastAPI application is running at `http://127.0.0.1:8000`.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.write("---")
st.info("Remember to keep your FastAPI backend running for this app to work!")

