import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# Load all BMP images from a folder
image_files = glob.glob("*.bmp")
images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]

histograms = [cv2.calcHist([img], [0], None, [256], [0,256]) for img in images]


# Stack histograms into an array
hist_array = np.array([h.flatten() for h in histograms])

# Plot average histogram
avg_hist = np.mean(hist_array, axis=0)
plt.plot(avg_hist)
plt.title("Average Histogram across 400 images")
plt.show()

mean_intensities = [np.mean(img) for img in images]

plt.plot(mean_intensities)
plt.title("Mean pixel intensity across images")
plt.xlabel("Image index")
plt.ylabel("Mean intensity")
plt.show()


# import cv2
# import glob
# import numpy as np
# import matplotlib.pyplot as plt

# # Load and sort image files
# image_files = sorted(glob.glob("*.bmp"))
# signal = []

# # Extract mean intensity from each image
# for file in image_files:
#     img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#     mean_val = np.mean(img)
#     signal.append(mean_val)

# # Plot the waveform
# plt.figure(figsize=(12, 6))
# plt.plot(signal, color='blue')
# plt.title("Seismograph-like Signal from Image Sequence")
# plt.xlabel("Frame Index")
# plt.ylabel("Mean Intensity (a.u.)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()