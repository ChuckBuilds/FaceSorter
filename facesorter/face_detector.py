import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class FaceDetector:
    """
    Handles face detection and facial feature extraction from images.
    """

    def __init__(self, model="hog"):
        """
        Initializes the FaceDetector.

        Args:
            model (str): The face detection model to use. Can be "hog" (less accurate, faster on CPU)
                         or "cnn" (more accurate, needs GPU/CUDA for speed).
        """
        self.model = model

    def detect_faces(self, image_path, min_face_area=None):
        """
        Detects faces in a single image, filters by area, and returns encodings, locations, and debug info.
        
        Args:
            image_path (str or Path): The path to the image file.
            min_face_area (int, optional): Minimum area for a detected face to be considered valid.
        Returns:
            A tuple containing:
            - image (np.array): The loaded image data.
            - face_encodings (list): A list of 128-dimensional encodings for each valid face.
            - face_locations (list): A list of bounding box coordinates for each valid face.
            - debug_info (list): A list of (location, area) tuples for ALL faces detected.
        """
        try:
            # Load the original image once
            original_image = face_recognition.load_image_file(image_path)
            
            # For both models, resize the image to be faster if it's large.
            # This reduces the amount of data to scan for faces.
            pil_image = Image.fromarray(original_image)
            max_size = 2048  
            if pil_image.height > max_size or pil_image.width > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            image_to_process = np.array(pil_image)
            
            # --- Performance Optimization ---
            # 1. Detect faces on the smaller, processed image.
            locations_on_processed = face_recognition.face_locations(image_to_process, model=self.model)

            # 2. Scale locations up to the original image size for accurate area filtering and cropping.
            locations_on_original = locations_on_processed
            if image_to_process.shape != original_image.shape:
                h_orig, w_orig, _ = original_image.shape
                h_proc, w_proc, _ = image_to_process.shape
                w_scale = w_orig / w_proc
                h_scale = h_orig / h_proc
                locations_on_original = [
                    (int(top * h_scale), int(right * w_scale), int(bottom * h_scale), int(left * w_scale))
                    for (top, right, bottom, left) in locations_on_processed
                ]

            # 3. Filter faces based on area, using the original scale for intuitive thresholds.
            debug_info = [(loc, (loc[1] - loc[3]) * (loc[2] - loc[0])) for loc in locations_on_original]

            valid_indices = list(range(len(locations_on_original)))
            if min_face_area:
                valid_indices = [i for i, (loc, area) in enumerate(debug_info) if area >= min_face_area]

            valid_locations_on_original = [locations_on_original[i] for i in valid_indices]
            
            # 4. Get encodings using the smaller `image_to_process` for a massive speedup.
            face_encodings = []
            if valid_indices:
                valid_locations_on_processed = [locations_on_processed[i] for i in valid_indices]
                face_encodings = face_recognition.face_encodings(image_to_process, known_face_locations=valid_locations_on_processed)

            # 5. Return the final data.
            return original_image, face_encodings, valid_locations_on_original, debug_info
                
        except Exception as e:
            # logger.error(f"Could not process image {image_path}: {e}")
            return None, [], [], []

    def get_face_encodings(self, image, face_locations):
        """
        Gets the 128-dimension face encoding for each face in the image.

        Args:
            image (np.array): The image data from load_image_file.
            face_locations (list): The list of face bounding boxes from face_locations.

        Returns:
            A list of face encodings (128-dimensional vectors).
        """
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return face_encodings

    @staticmethod
    def draw_face_tags(image, face_locations_with_areas):
        """
        Draws bounding boxes and numbers for each face to help with debugging.

        Args:
            image (np.array): The image data.
            face_locations_with_areas (list): A list of (location, area) tuples from detect_faces.

        Returns:
            A PIL Image object with tagged faces.
        """
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        try:
            # Use a default font. A specific .ttf file could be provided for better results.
            font = ImageFont.load_default()
        except IOError:
            font = None

        for i, (loc, area) in enumerate(face_locations_with_areas):
            top, right, bottom, left = loc
            # Draw rectangle
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
            
            # Prepare text and background
            text = f"Face {i+1}"
            
            # Draw a filled rectangle as a background for the text for better visibility
            if font:
                # Pillow 9.2.0+ uses textbbox for more accurate size calculation
                try:
                    text_bbox = draw.textbbox((left, top), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    bg_y0 = top - text_height - 10
                    # Adjust so text is above the box, not overlapping
                    if bg_y0 < 0:
                        bg_y0 = bottom + 10
                    
                    draw.rectangle(((left, bg_y0), (left + text_width + 8, bg_y0 + text_height + 4)), fill=(0, 255, 0))
                    draw.text((left + 4, bg_y0 + 2), text, fill=(0, 0, 0), font=font)
                except AttributeError:
                    # Fallback for older Pillow versions
                    text_width, text_height = draw.textsize(text, font=font)
                    bg_y0 = top - text_height - 10
                    if bg_y0 < 0:
                        bg_y0 = bottom + 10
                    draw.rectangle(((left, bg_y0), (left + text_width + 8, bg_y0 + text_height + 4)), fill=(0, 255, 0))
                    draw.text((left + 4, bg_y0 + 2), text, fill=(0, 0, 0), font=font)
            else:
                 # Fallback if no font is available
                 draw.text((left + 4, top - 15), text, fill=(0, 255, 0))

        return pil_image

    @staticmethod
    def draw_diagnostic_tags(image, face_info_list, min_area_threshold):
        """
        Draws bounding boxes and detailed diagnostic info for each face.

        Args:
            image (np.array): The image data.
            face_info_list (list): A list of (location, area) tuples from detect_faces debug_info.
            min_area_threshold (int): The minimum area to highlight which faces are kept.

        Returns:
            A PIL Image object with tagged faces.
        """
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        try:
            # Use a default font. A specific .ttf file could be provided for better results.
            font = ImageFont.load_default(60)
        except IOError:
            font = None

        for i, (loc, area) in enumerate(face_info_list):
            top, right, bottom, left = loc
            
            is_valid = area >= min_area_threshold
            box_color = (0, 255, 0) if is_valid else (255, 0, 0) # Green for valid, Red for invalid
            
            # Draw rectangle
            draw.rectangle(((left, top), (right, bottom)), outline=box_color, width=5)
            
            # Prepare text and background
            text = f"Face {i+1}\nArea: {int(area)}"
            
            # Draw a filled rectangle as a background for the text for better visibility
            if font:
                # Pillow 9.2.0+ uses textbbox for more accurate size calculation
                try:
                    text_bbox = draw.textbbox((left, top), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    bg_y0 = top - text_height - 10
                    # Adjust so text is above the box, not overlapping
                    if bg_y0 < 0:
                        bg_y0 = bottom + 10
                    
                    draw.rectangle(((left, bg_y0), (left + text_width + 8, bg_y0 + text_height + 4)), fill=box_color)
                    draw.text((left + 4, bg_y0 + 2), text, fill=(0, 0, 0), font=font)
                except AttributeError:
                    # Fallback for older Pillow versions
                    text_width, text_height = draw.textsize(text, font=font)
                    bg_y0 = top - text_height - 10
                    if bg_y0 < 0:
                        bg_y0 = bottom + 10
                    draw.rectangle(((left, bg_y0), (left + text_width + 8, bg_y0 + text_height + 4)), fill=box_color)
                    draw.text((left + 4, bg_y0 + 2), text, fill=(0, 0, 0), font=font)
            else:
                 # Fallback if no font is available
                 draw.text((left + 4, top - 15), text, fill=box_color)

        return pil_image

    @staticmethod
    def draw_faces(image, face_locations):
        """
        Draws bounding boxes around the detected faces.

        Args:
            image (np.array): The image data.
            face_locations (list): A list of face bounding box coordinates.

        Returns:
            A PIL Image object with rectangles drawn around the faces.
        """
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        for (top, right, bottom, left) in face_locations:
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)

        return pil_image

    @staticmethod
    def crop_faces(image, face_locations, padding=30):
        """
        Crops faces from an image using the bounding box coordinates.

        Args:
            image (np.array): The image data.
            face_locations (list): A list of face bounding box coordinates.
            padding (int): The number of pixels to add around the detected face box.

        Returns:
            A list of PIL Image objects, each being a cropped face.
        """
        pil_image = Image.fromarray(image)
        face_crops = []
        img_height, img_width, _ = image.shape

        for (top, right, bottom, left) in face_locations:
            # Add padding and ensure coordinates are within image bounds
            top = max(0, top - padding)
            left = max(0, left - padding)
            right = min(img_width, right + padding)
            bottom = min(img_height, bottom + padding)
            
            # Crop the face from the image
            face_image = pil_image.crop((left, top, right, bottom))
            face_crops.append(face_image)

        return face_crops 

    def get_face_locations(self, image_path, model="hog"):
        """Returns the locations of faces in an image."""
        try:
            image = face_recognition.load_image_file(image_path)
            return face_recognition.face_locations(image, model=model)
        except Exception as e:
            # logger.error(f"Could not get face locations from {image_path}: {e}")
            return []

    def extract_face_crops(self, image_path, face_locations):
        """Extracts and returns image crops for each detected face."""
        if not face_locations:
            return []
            
        try:
            image = face_recognition.load_image_file(image_path)
            face_crops = []
            for top, right, bottom, left in face_locations:
                face_crops.append(image[top:bottom, left:right])
            return face_crops
        except Exception as e:
            # logger.error(f"Could not extract face crops from {image_path}: {e}")
            return []