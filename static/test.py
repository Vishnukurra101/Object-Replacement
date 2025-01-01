import cv2

# Read the image in BGR format
img = cv2.imread(r"C:\Users\vishn\OneDrive\Documents\codes\Image_Pipeline\Object-Replacement\static\result.jpg")

# Display the original BGR image
cv2.imshow("Original BGR Image", img)
cv2.waitKey(0)

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the converted RGB image
cv2.imshow("Converted RGB Image", img_rgb)
cv2.waitKey(0)

# Convert back to BGR (which is how OpenCV typically handles images)
img_bgr_again = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Display the converted-back BGR image
cv2.imshow("Converted Back to BGR", img_bgr_again)
cv2.waitKey(0)

cv2.destroyAllWindows()