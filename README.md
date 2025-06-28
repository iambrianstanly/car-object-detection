<<<<<<< HEAD
# car-object-detection
detecting cars on road
=======
# Car Object Detection Web Application

## Project Description
This project implements a simple yet effective Car Object Detection API using FastAPI. It allows users to upload an image, and the API will process it with a pre-trained TensorFlow model to determine if a car is present, predict its bounding box

## Features
- **Car Detection:** Utilizes a trained deep learning model to identify cars in various images.

- **Bounding Box Localization:** Draws bounding boxes around detected cars, providing their precise location.

- **FastAPI Backend:** A high-performance Python web API built with FastAPI.

- **Docker Containerization:** Ensures a consistent and isolated environment for the application, making it highly portable and easy to deploy.

- **User-Friendly Interface (Implicit):** Designed to accept image inputs and return processed images or detection data.

- **Scalable:** Docker containerization allows for easy scaling of the application.


## Technologies Used

- **Deep Learning Framework:** Tensorflow, keras

- **Deep Learning Model:** Xception model , used transfer learning concept.

- **Backend Framework:** FastAPI

- **Containerization:** Docker

- **Programming Language:** Python 3.10.12


## Usage

The application exposes a RESTful API for car detection. You can interact with it using the Swagger UI (at http://localhost:8000/docs) or by sending HTTP requests directly.

### Example: Detecting Cars from an Image
To detect cars in an image, you will typically send a POST request to the /detect (or similar) endpoint, uploading an image file.

**Via Swagger UI:**

- Go to http://localhost:8000/docs.

- Find the /predict endpoint.

- Click "Try it out".

- Upload an image file using the file input field.

- Click "Execute".

The response will contain either the class labels and  bounding boxes in JSON data detailing the detections.
>>>>>>> f0fd379 (changed)
