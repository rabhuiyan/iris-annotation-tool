import cv2
import numpy as np
import os


class ImageAnnotator:
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path
        self.current_points = []  # Points of the current polygon
        self.completed_polygons = []  # All completed polygons
        self.mask = None  # Combined mask for all shapes
        self.image = cv2.imread(image_path)
        self.padding = 10
        self.padded_image = cv2.copyMakeBorder(self.image, self.padding, self.padding, self.padding, self.padding,
                                               cv2.BORDER_CONSTANT)
        self.mask = np.zeros(self.padded_image.shape[:2], dtype=np.uint8)

    def draw_polygon(self, event, x, y, flags, param):
        """Capture points of the polygon using mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))

    def show_image_with_polygon(self):
        """Display the image with the current polygon outline and completed polygons."""
        temp_image = self.padded_image.copy()

        # Draw all completed polygons in red
        if self.completed_polygons:
            cv2.polylines(temp_image, [np.array(polygon, dtype=np.int32) for polygon in self.completed_polygons],
                          isClosed=True, color=(0, 0, 255), thickness=1)

        # Draw the current polygon outline in green if there are points
        if len(self.current_points) > 1:
            cv2.polylines(temp_image, [np.array(self.current_points, dtype=np.int32)], isClosed=False,
                          color=(0, 255, 0), thickness=1)

        cv2.imshow("Annotate Image", temp_image)

    def save_mask(self):
        """Save the combined mask to the specified output path."""
        if self.completed_polygons:  # Only save if there are completed polygons
            mask = self.mask[self.padding:-self.padding, self.padding:-self.padding]  # Remove padding
            image_name = os.path.basename(self.image_path)
            mask_output_path = os.path.join(self.output_path, image_name)
            cv2.imwrite(mask_output_path, mask)
            print(f"Mask saved to {mask_output_path}")
        else:
            print("No shapes drawn. Mask not saved.")

    def annotate(self):
        """Annotate the image to create a single binary mask for all shapes."""
        cv2.namedWindow("Annotate Image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotate Image", self.draw_polygon)

        while True:
            self.show_image_with_polygon()
            key = cv2.waitKey(1) & 0xFF

            # Complete the current polygon
            if key == ord('c') and len(self.current_points) > 2:
                adjusted_polygon = [(x, y) for (x, y) in self.current_points]
                cv2.fillPoly(self.mask, [np.array(adjusted_polygon, dtype=np.int32)], 255)
                self.completed_polygons.append(adjusted_polygon)
                print("Polygon added to the mask. Draw the next shape or press 's' to save.")
                self.current_points.clear()  # Reset points for the next shape

            # Save the mask and exit
            elif key == ord('s'):
                break

        cv2.destroyAllWindows()
        self.save_mask()


def save_progress(index, progress_file):
    with open(progress_file, 'w') as f:
        f.write(str(index))


def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return int(f.read().strip())
    return 0  # If no progress file exists, start from 0


if __name__ == "__main__":
    # Define the image path and output path
    image_dir = '../warsaw-nij-original-nir-images'
    output_path = './all-masks/'
    progress_file = './annotation_progress.txt'  # File to track progress

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load the last processed index
    if os.path.exists(progress_file):
        start_index = load_progress(progress_file)
    else:
        start_index = 0

    filenames = os.listdir(image_dir)

    for i, filename in enumerate(filenames[start_index:], start=start_index):
        image_path = os.path.join(image_dir, filename)

        print(f"{i}/{len(filenames)}:: {filename}")

        # Create an annotator object and run the annotation function
        annotator = ImageAnnotator(image_path, output_path)
        annotator.annotate()

        # Save the progress after processing each image
        save_progress(i + 1, progress_file)
