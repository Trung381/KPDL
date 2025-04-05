#________________________________ tô màumàu
# # to mau
# import cv2
# import numpy as np
# from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt

# def segment_image_dbscan_optimized(image_path, eps=5, min_samples=10):
#     # Đọc ảnh, resize nhỏ lại để giảm số điểm
#     image = cv2.imread(image_path)
#     image_small = cv2.resize(image, (100, 100))  # Resize về 100x100
#     lab_image = cv2.cvtColor(image_small, cv2.COLOR_BGR2LAB)

#     h, w, _ = lab_image.shape
#     features = []

#     # Tạo vector gồm L, A, B + tọa độ x, y
#     for y in range(h):
#         for x in range(w):
#             l, a, b = lab_image[y, x]
#             features.append([l, a, b, x, y])

#     features = np.array(features)

#     # Áp dụng DBSCAN
#     dbscan = DBSCAN(eps=5, min_samples=min_samples)
#     labels = dbscan.fit_predict(features)

#     # Gán màu theo nhãn
#     unique_labels = np.unique(labels)
#     colors = np.random.randint(0, 255, size=(len(unique_labels), 3))
#     segmented = np.zeros((h * w, 3), dtype=np.uint8)

#     for idx, label in enumerate(labels):
#         if label == -1:
#             segmented[idx] = [0, 0, 0]
#         else:
#             segmented[idx] = colors[label]

#     # Biến về ảnh và resize lại kích thước gốc
#     segmented_img = segmented.reshape((h, w, 3))
#     segmented_img = cv2.resize(segmented_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

#     # Hiển thị
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title("Original")

#     plt.subplot(1, 2, 2)
#     plt.imshow(segmented_img)
#     plt.title("Segmented (DBSCAN)")

#     plt.show()

# # Gọi hàm
# segment_image_dbscan_optimized('Tr-pi_0751.jpg')

#____________________________ nhìn chả khác mẹ j thêm filter cho ảnhảnh

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import DBSCAN

# img = cv2.imread('Tr-pi_0751.jpg')

# Z = np.float32(img.reshape((-1,3)))
# db = DBSCAN(eps=0.3, min_samples=100).fit(Z[:,:2])

# plt.imshow(np.uint8(db.labels_.reshape(img.shape[:2])))
# plt.show()

#___________________________ cái thêm filter nhưng là kmeankmean

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import DBSCAN

# img = cv2.imread('Tr-pi_0751.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Z = np.float32(img.reshape((-1,3)))

# # Define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ret, label, center = cv2.kmeans(Z, 5, None, criteria, 6, cv2.KMEANS_RANDOM_CENTERS)

# # Now convert back into uint8, and make the original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = np.uint8(label.reshape(img.shape[:2]))
# res2.shape

# plt.imshow(res2)
# plt.show()


#__________________________ gemini

# import cv2
# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# def segment_tumor_dbscan(image_path, eps=1.5, min_samples=50, downscale_factor=1.0):
#     """
#     Segments an MRI image using DBSCAN based on intensity and spatial location.

#     Args:
#         image_path (str): Path to the MRI image file.
#         eps (float): The maximum distance between two samples for one to be
#                      considered as in the neighborhood of the other (DBSCAN parameter).
#                      Needs careful tuning based on scaled features.
#         min_samples (int): The number of samples in a neighborhood for a point
#                            to be considered as a core point (DBSCAN parameter).
#                            Needs careful tuning.
#         downscale_factor (float): Factor to downscale the image by (e.g., 0.5 for half size).
#                                   Reduces computation time but loses detail. 1.0 means no downscaling.

#     Returns:
#         tuple: (original_image, clustered_image, potential_tumor_mask, labels) or None if image fails to load.
#                - original_image: The loaded (and potentially resized) grayscale image.
#                - clustered_image: Image showing all found clusters with different colors.
#                - potential_tumor_mask: Binary mask highlighting the most likely tumor cluster.
#                - labels: Raw cluster labels assigned by DBSCAN (-1 for noise).
#     """
#     # 1. Load Image
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"Error: Could not load image from {image_path}")
#         return None

#     original_shape = img.shape
#     print(f"Original image shape: {original_shape}")

#     # 2. Preprocessing - Downscaling (Optional)
#     if downscale_factor != 1.0:
#         new_width = int(img.shape[1] * downscale_factor)
#         new_height = int(img.shape[0] * downscale_factor)
#         img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
#         print(f"Resized image shape: {img.shape}")

#     h, w = img.shape

#     # 3. Prepare Feature Matrix
#     # Create coordinate grids
#     y_coords, x_coords = np.indices(img.shape) # Creates h x w arrays of y and x coords

#     # Flatten image and coordinates
#     intensity = img.flatten()
#     x_coords_flat = x_coords.flatten()
#     y_coords_flat = y_coords.flatten()

#     # Combine features: [intensity, x_coord, y_coord]
#     # We weight coordinates less if needed, but start with equal weighting via scaling
#     features = np.vstack((intensity, x_coords_flat, y_coords_flat)).T # Transpose to get (n_pixels, 3)

#     print(f"Feature matrix shape: {features.shape}")

#     # --- Important Check: Handle completely black/white images ---
#     if np.std(features[:, 0]) == 0: # Check if intensity has zero standard deviation
#          print("Warning: Image intensity is uniform. DBSCAN might not produce meaningful results.")
#          # Optionally return or proceed with caution
#          # return img, np.zeros_like(img), np.zeros_like(img), np.zeros(img.size) -1


#     # 4. Feature Scaling
#     print("Scaling features...")
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)
#     print("Features scaled.")

#     # 5. Apply DBSCAN
#     print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
#     db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1) # Use all available CPU cores
#     db.fit(scaled_features)
#     labels = db.labels_ # Cluster labels for each point (-1 is noise)
#     print("DBSCAN complete.")

#     # Get number of clusters (excluding noise)
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#     print(f'Estimated number of clusters: {n_clusters_}')
#     print(f'Estimated number of noise points: {n_noise_}')

#     # 6. Reshape Labels to Image Dimensions
#     clustered_label_img = labels.reshape(h, w)

#     # 7. Visualize All Clusters
#     # Create an image where each cluster has a unique color
#     # Map noise (-1) to black (0) and clusters (0, 1, 2...) to distinct grayscale values or colors
#     unique_labels = np.unique(labels)
#     clustered_image_viz = np.zeros((h, w, 3), dtype=np.uint8) # Create a color image

#     # Generate colors (excluding black for noise)
#     colors = plt.cm.get_cmap('viridis', n_clusters_ + 1) # Using viridis colormap

#     for label_value in unique_labels:
#         if label_value == -1:
#             # Noise points - color them black or gray
#             clustered_image_viz[clustered_label_img == label_value] = [50, 50, 50] # Dark Gray
#         else:
#             # Assign a unique color to each cluster
#             cluster_color = (np.array(colors(label_value)[:3]) * 255).astype(np.uint8)
#             clustered_image_viz[clustered_label_img == label_value] = cluster_color

#     # 8. Identify Potential Tumor Cluster (Heuristic - NEEDS REFINEMENT)
#     # This is a very basic heuristic. Real-world scenarios need better logic.
#     # Assumptions: Tumor is not the largest cluster (background) and not noise.
#     #              Tumor might have a relatively high average intensity (or low, depending on MRI type).
#     potential_tumor_mask = np.zeros((h, w), dtype=np.uint8)
#     best_cluster_label = -1
#     min_cluster_size_threshold = int(0.001 * h * w) # Ignore very small clusters (e.g., < 0.1% of pixels)
#     max_cluster_size_threshold = int(0.5 * h * w) # Ignore very large clusters (likely background)

#     cluster_properties = []
#     for label_value in unique_labels:
#         if label_value == -1:
#             continue # Skip noise

#         cluster_mask = (clustered_label_img == label_value)
#         cluster_size = np.sum(cluster_mask)
#         avg_intensity = np.mean(img[cluster_mask])

#         # --- Filtering Logic (Example) ---
#         if min_cluster_size_threshold < cluster_size < max_cluster_size_threshold:
#              # Calculate centroid
#             y_indices, x_indices = np.where(cluster_mask)
#             centroid_y, centroid_x = np.mean(y_indices), np.mean(x_indices)

#             # Store properties for potential later analysis/selection
#             cluster_properties.append({
#                 'label': label_value,
#                 'size': cluster_size,
#                 'avg_intensity': avg_intensity,
#                 'centroid': (centroid_y, centroid_x)
#             })
#             print(f"  Cluster {label_value}: Size={cluster_size}, AvgIntensity={avg_intensity:.2f}, Centroid=({centroid_y:.1f}, {centroid_x:.1f})")


#     # --- Selection Logic (Example: Choose cluster closest to center with intensity above median) ---
#     if cluster_properties:
#         center_y, center_x = h / 2, w / 2
#         median_intensity = np.median(img) # Overall median intensity

#         best_candidate = None
#         min_dist_to_center = float('inf')

#         # Sort by intensity (descending) as a simple proxy, might need adjustment
#         cluster_properties.sort(key=lambda p: p['avg_intensity'], reverse=True)

#         # Example: Select the highest intensity cluster within size bounds
#         if cluster_properties: # Check if any clusters met the size criteria
#              best_cluster_label = cluster_properties[0]['label'] # Take the highest intensity one passing size filter
#              print(f"Selected Cluster Label (Heuristic): {best_cluster_label}")

#         # # Alternative: Select cluster closest to the center (example)
#         # for prop in cluster_properties:
#         #     # Optional: Add intensity check if needed (e.g., intensity > median_intensity * 1.1)
#         #     # if prop['avg_intensity'] < median_intensity * 1.1: # Example: must be brighter than median
#         #     #    continue
#         #     cy, cx = prop['centroid']
#         #     dist = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
#         #     if dist < min_dist_to_center:
#         #         min_dist_to_center = dist
#         #         best_cluster_label = prop['label']


#     if best_cluster_label != -1:
#         potential_tumor_mask[clustered_label_img == best_cluster_label] = 255


#     # 9. Resize results back if downscaled
#     if downscale_factor != 1.0:
#          img = cv2.resize(img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
#          clustered_image_viz = cv2.resize(clustered_image_viz, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
#          potential_tumor_mask = cv2.resize(potential_tumor_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
#          # Reshape labels back requires more care, maybe skip resizing raw labels if not needed later
#          # labels_resized = cv2.resize(clustered_label_img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST).flatten()


#     return img, clustered_image_viz, potential_tumor_mask, labels # Return original shape results

# # --- Example Usage ---
# if __name__ == "__main__":
#     # Replace with the actual path to your MRI image
#     image_path = 'licensed-image.jpg' # Or .jpg, .tif, .dcm (install pydicom for DICOM)

#     # --- TUNABLE PARAMETERS ---
#     # These HIGHLY depend on your image properties and the effect of scaling.
#     # Start with values like these and adjust.
#     # Increase eps to merge more points, decrease to make clusters tighter.
#     # Increase min_samples to make clusters denser and create more noise points.
#     dbscan_eps = 1.5  # Adjust based on observation (distance in scaled feature space)
#     dbscan_min_samples = 75 # Adjust based on expected tumor size/density
#     downscale = 0.5 # Use 0.5 for half size, 1.0 for full size (slower)
#     # ---

#     results = segment_tumor_dbscan(image_path,
#                                    eps=dbscan_eps,
#                                    min_samples=dbscan_min_samples,
#                                    downscale_factor=downscale)

#     if results:
#         original_image, clustered_image, potential_tumor_mask, _ = results

#         # --- Visualization ---
#         plt.figure(figsize=(18, 6))

#         plt.subplot(1, 3, 1)
#         plt.imshow(original_image, cmap='gray')
#         plt.title('Original MRI Image')
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.imshow(clustered_image) # Already in color
#         plt.title(f'DBSCAN Clustering (eps={dbscan_eps}, min={dbscan_min_samples})')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.imshow(original_image, cmap='gray')
#         plt.imshow(potential_tumor_mask, cmap='jet', alpha=0.5) # Overlay mask
#         plt.title('Potential Tumor Mask (Heuristic)')
#         plt.axis('off')

#         plt.tight_layout()
#         plt.show()

#         # Optionally save the results
#         # cv2.imwrite('clustered_output.png', clustered_image)
#         # cv2.imwrite('tumor_mask_output.png', potential_tumor_mask)

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time # Thêm để đo thời gian xử lý

# Tùy chọn: Thêm thư viện để đọc DICOM nếu cần
# import pydicom
# from pydicom.pixel_data_handlers.util import apply_voi_lut

def segment_tumor_dbscan(image_path, eps=1.5, min_samples=50, downscale_factor=0.25): # Đặt giá trị mặc định nhỏ hơn (0.25 là 1/4 kích thước)
    """
    Phân đoạn ảnh MRI sử dụng DBSCAN dựa trên cường độ và vị trí không gian.
    Tối ưu hóa cho ảnh lớn bằng cách giảm kích thước trước khi xử lý DBSCAN.

    Args:
        image_path (str): Đường dẫn đến tệp ảnh MRI (PNG, JPG, TIF, hoặc DCM nếu có pydicom).
        eps (float): Khoảng cách tối đa giữa hai mẫu để được coi là lân cận (tham số DBSCAN).
                     !!! QUAN TRỌNG: Cần điều chỉnh kỹ lưỡng sau khi thay đổi downscale_factor !!!
                     Tham số này nằm trong không gian đặc trưng đã được scale.
        min_samples (int): Số lượng mẫu tối thiểu trong vùng lân cận của một điểm để điểm đó
                           được coi là điểm lõi (tham số DBSCAN).
                           !!! QUAN TRỌNG: Cần điều chỉnh kỹ lưỡng sau khi thay đổi downscale_factor !!!
                           Ảnh hưởng đến mật độ cụm yêu cầu.
        downscale_factor (float): Hệ số giảm kích thước ảnh (ví dụ: 0.5 cho nửa kích thước, 0.25 cho 1/4).
                                  Giá trị trong khoảng (0, 1]. 1.0 là kích thước gốc (có thể rất chậm với ảnh lớn).
                                  Việc giảm kích thước giúp giảm đáng kể thời gian tính toán và bộ nhớ sử dụng.

    Returns:
        tuple: (original_image, clustered_image_final, potential_tumor_mask_final, labels) hoặc None nếu có lỗi.
               - original_image: Ảnh gốc (grayscale, kích thước ban đầu).
               - clustered_image_final: Ảnh màu hiển thị các cụm khác nhau (đã resize về kích thước ban đầu).
               - potential_tumor_mask_final: Mặt nạ nhị phân (0 hoặc 255) đánh dấu cụm có khả năng là khối u (đã resize về kích thước ban đầu).
               - labels: Mảng 1D chứa nhãn cụm thô (-1 là nhiễu) được gán bởi DBSCAN cho các pixel TRONG ẢNH ĐÃ RESIZE.
                         Kích thước là (số_pixel_sau_resize,). Không được resize về kích thước gốc.
    """
    start_time = time.time()

    # 1. Load Image
    print(f"Loading image: {image_path}...")
    img = None
    # --- Xử lý DICOM (Nếu đường dẫn kết thúc bằng .dcm) ---
    if image_path.lower().endswith(".dcm"):
        try:
            # Chỉ import nếu cần thiết
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut

            dicom_data = pydicom.dcmread(image_path)
            # Lấy dữ liệu pixel và áp dụng VOI LUT (quan trọng để có dải cường độ đúng)
            img_array = apply_voi_lut(dicom_data.pixel_array, dicom_data)

            # Chuẩn hóa về uint8 (0-255) để xử lý nhất quán
            # Tránh chia cho 0 nếu ảnh hoàn toàn đen hoặc trắng
            min_val = np.min(img_array)
            max_val = np.max(img_array)
            if max_val > min_val:
                img = ((img_array - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            else: # Ảnh đồng nhất
                img = np.full(img_array.shape, int(min_val) if min_val <= 255 else 255 , dtype=np.uint8)

            # Đảm bảo là ảnh xám (một số DICOM có thể có kênh màu)
            if len(img.shape) > 2 and img.shape[2] >= 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 2:
                 pass # Đã là ảnh xám
            else:
                 print(f"Warning: Unexpected DICOM image shape after load: {img.shape}")
                 # Có thể cần xử lý thêm tùy định dạng cụ thể

            print("DICOM image loaded and converted to grayscale uint8 successfully.")

        except ImportError:
            print("\nError: Thư viện 'pydicom' chưa được cài đặt.")
            print("Vui lòng cài đặt bằng lệnh: pip install pydicom")
            print("Không thể đọc file DICOM.\n")
            return None
        except Exception as e:
            print(f"Error reading or processing DICOM file {image_path}: {e}")
            return None
    else:
        # --- Xử lý ảnh thông thường (PNG, JPG, TIF, ...) ---
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra nếu ảnh không tải được
    if img is None:
        print(f"Error: Không thể tải ảnh từ đường dẫn {image_path}")
        return None

    original_shape = img.shape
    print(f"Original image shape: {original_shape}")

    # --- KIỂM TRA KÍCH THƯỚC VÀ CẢNH BÁO ---
    # Ảnh lớn hơn 1 Megapixel và không giảm kích thước
    if original_shape[0] * original_shape[1] > 1024 * 1024 and downscale_factor == 1.0:
         print("\n*** CẢNH BÁO ***")
         print("Ảnh có kích thước lớn và bạn đang xử lý ở kích thước gốc (downscale_factor=1.0).")
         print("Quá trình này có thể RẤT CHẬM hoặc làm MÁY BỊ TREO do thiếu RAM.")
         print(">>> Nên đặt 'downscale_factor' thành giá trị nhỏ hơn (ví dụ: 0.5, 0.25) để tăng tốc độ.")
         print("-" * 15)

    # 2. Preprocessing - Downscaling (Giảm kích thước ảnh để xử lý)
    img_processed = img.copy() # Làm việc trên bản sao đã resize
    if downscale_factor != 1.0:
        # Kiểm tra giá trị hợp lệ
        if not (0 < downscale_factor <= 1.0):
             print(f"Warning: downscale_factor ({downscale_factor}) không hợp lệ. Sử dụng 1.0 (không giảm kích thước).")
             downscale_factor = 1.0
        else:
            new_width = int(img_processed.shape[1] * downscale_factor)
            new_height = int(img_processed.shape[0] * downscale_factor)
            # Đảm bảo kích thước tối thiểu là 1 pixel
            new_width = max(1, new_width)
            new_height = max(1, new_height)

            # Sử dụng INTER_AREA cho việc thu nhỏ để có kết quả tốt hơn
            img_processed = cv2.resize(img_processed, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Resized image shape for processing: {img_processed.shape}")
    else:
        print("Processing at original size.")


    h, w = img_processed.shape # Kích thước ảnh dùng để xử lý DBSCAN

    # 3. Prepare Feature Matrix (Chuẩn bị ma trận đặc trưng)
    # Đặc trưng bao gồm: cường độ pixel, tọa độ x, tọa độ y
    print("Preparing feature matrix...")
    y_coords, x_coords = np.indices(img_processed.shape) # Tạo lưới tọa độ y, x

    intensity = img_processed.flatten()      # Làm phẳng ảnh thành mảng 1D cường độ
    x_coords_flat = x_coords.flatten()       # Làm phẳng lưới tọa độ x
    y_coords_flat = y_coords.flatten()       # Làm phẳng lưới tọa độ y

    # Ghép các đặc trưng lại thành ma trận: mỗi hàng là một pixel, 3 cột là [cường độ, x, y]
    features = np.vstack((intensity, x_coords_flat, y_coords_flat)).T
    print(f"Feature matrix shape: {features.shape} (pixels, features)")

    # --- Kiểm tra ảnh có cường độ đồng nhất không ---
    # Nếu độ lệch chuẩn của cường độ quá nhỏ, DBSCAN sẽ không hiệu quả
    if np.std(features[:, 0]) < 1e-6: # So sánh với một giá trị rất nhỏ
         print("\nWarning: Cường độ ảnh (sau khi resize) gần như đồng nhất.")
         print("DBSCAN có thể không tạo ra kết quả phân cụm ý nghĩa. Trả về kết quả rỗng.\n")
         # Trả về ảnh gốc và các mask/ảnh cluster rỗng hoặc chỉ có nhiễu
         clustered_img_viz_final = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Ảnh gốc chuyển sang màu
         potential_tumor_mask_final = np.zeros(original_shape, dtype=np.uint8) # Mask đen
         labels_dummy = np.full(features.shape[0], -1, dtype=int) # Tất cả pixel là nhiễu (-1)
         return img, clustered_img_viz_final, potential_tumor_mask_final, labels_dummy


    # 4. Feature Scaling (Chuẩn hóa đặc trưng)
    # Rất quan trọng để DBSCAN hoạt động đúng vì nó dựa trên khoảng cách.
    # StandardScaler làm cho mỗi đặc trưng (cường độ, x, y) có trung bình 0 và phương sai 1.
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("Features scaled.")
    scaling_time = time.time()
    print(f"--> Time for Load/Resize/FeaturePrep/Scale: {scaling_time - start_time:.2f} seconds")


    # 5. Apply DBSCAN (Áp dụng thuật toán DBSCAN)
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    # Sử dụng n_jobs=-1 để dùng tất cả các lõi CPU có sẵn, tăng tốc độ đáng kể
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    db.fit(scaled_features) # Huấn luyện (thực ra là chạy) DBSCAN trên dữ liệu đã scale
    labels = db.labels_ # Lấy nhãn cụm cho từng pixel (điểm dữ liệu)
                        # labels là mảng 1D, cùng kích thước với số hàng của scaled_features
                        # Giá trị: 0, 1, 2,... cho các cụm; -1 cho các điểm nhiễu (noise)
    dbscan_time = time.time()
    print("DBSCAN complete.")
    print(f"--> Time for DBSCAN fitting: {dbscan_time - scaling_time:.2f} seconds")


    # --- Thống kê kết quả DBSCAN ---
    unique_labels_set = set(labels) # Lấy các nhãn duy nhất
    n_clusters_ = len(unique_labels_set) - (1 if -1 in unique_labels_set else 0) # Đếm số cụm (trừ nhiễu -1)
    n_noise_ = np.count_nonzero(labels == -1) # Đếm số điểm nhiễu

    print(f'Estimated number of clusters found: {n_clusters_}')
    if len(labels) > 0:
        print(f'Estimated number of noise points: {n_noise_} / {len(labels)} ({n_noise_*100.0/len(labels):.1f}%)')
    else:
        print('No data points processed.') # Trường hợp rất hiếm

    # Xử lý trường hợp không tìm thấy cụm nào (chỉ có nhiễu)
    if n_clusters_ == 0 and n_noise_ > 0:
        print("\nWarning: DBSCAN found NO clusters, only noise points.")
        print("Consider adjusting parameters:")
        print("  - Increase 'eps' to allow points further apart to connect.")
        print("  - Decrease 'min_samples' to require fewer points to form a dense region.")
        # Trả về kết quả rỗng/noise tương tự trường hợp ảnh đồng nhất
        clustered_img_viz_final = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        potential_tumor_mask_final = np.zeros(original_shape, dtype=np.uint8)
        return img, clustered_img_viz_final, potential_tumor_mask_final, labels


    # 6. Reshape Labels to Processed Image Dimensions (Định hình lại nhãn về kích thước ảnh đã xử lý)
    # Chuyển mảng nhãn 1D thành ảnh 2D có cùng kích thước với img_processed
    clustered_label_img = labels.reshape(h, w)

    # 7. Visualize All Clusters (Tạo ảnh màu để hiển thị các cụm)
    # Sử dụng kích thước ảnh đã xử lý (h, w)
    print("Visualizing clusters...")
    clustered_image_viz = np.zeros((h, w, 3), dtype=np.uint8) # Tạo ảnh đen 3 kênh (màu)
    unique_labels_viz = np.unique(labels) # Lấy lại các nhãn duy nhất để lặp qua

    # Tạo một bảng màu (colormap) để gán màu khác nhau cho mỗi cụm
    # +1 để có màu riêng cho nhiễu nếu muốn, hoặc chỉ dùng cho các cụm hợp lệ
    if n_clusters_ > 0:
         # Dùng colormap 'viridis' hoặc 'jet', 'hsv',...
         colors = plt.cm.get_cmap('viridis', n_clusters_)
    else:
         colors = None # Không có cụm để tô màu


    cluster_color_map = {} # Lưu map từ label sang màu để nhất quán
    color_index = 0
    for label_value in unique_labels_viz:
        # Xác định các pixel thuộc về nhãn hiện tại
        mask = (clustered_label_img == label_value)

        if label_value == -1:
            # Gán màu xám đậm cho nhiễu
            cluster_color_map[label_value] = [50, 50, 50] # Dark Gray
        else:
            # Lấy màu từ colormap cho các cụm hợp lệ (0, 1, 2, ...)
            if colors:
                 # Lấy giá trị màu RGBA (0-1), bỏ kênh Alpha, chuyển về RGB 0-255
                 color_rgb = (np.array(colors(color_index)[:3]) * 255).astype(np.uint8)
                 cluster_color_map[label_value] = color_rgb
                 color_index += 1
            else: # Trường hợp không có cluster nào cả
                 cluster_color_map[label_value] = [0,0,0] # Màu đen

        # Tô màu các pixel tương ứng trên ảnh visualization
        clustered_image_viz[mask] = cluster_color_map[label_value]


    # 8. Identify Potential Tumor Cluster (Xác định cụm có khả năng là khối u - Dùng Heuristic)
    # !!! PHẦN NÀY MANG TÍNH GIẢ ĐỊNH CAO VÀ CẦN ĐƯỢC TINH CHỈNH CHO DỮ LIỆU CỤ THỂ !!!
    print("Identifying potential tumor cluster (using heuristics)...")
    potential_tumor_mask = np.zeros((h, w), dtype=np.uint8) # Tạo mask đen kích thước xử lý
    best_cluster_label = -1 # Nhãn của cụm được chọn, mặc định là không có

    # --- Định nghĩa các ngưỡng lọc cụm ---
    # Ngưỡng kích thước (tính theo % số pixel của ảnh ĐÃ RESIZE)
    # Loại bỏ các cụm quá nhỏ (nhiễu?) hoặc quá lớn (nền?)
    # Điều chỉnh các ngưỡng này dựa trên kích thước tương đối dự kiến của khối u
    min_cluster_size_ratio = 0.0005  # Tối thiểu 0.05% số pixel
    max_cluster_size_ratio = 0.30    # Tối đa 30% số pixel
    min_cluster_size_threshold = max(15, int(min_cluster_size_ratio * h * w)) # Đảm bảo ít nhất 15 pixel
    max_cluster_size_threshold = int(max_cluster_size_ratio * h * w)
    print(f"Cluster size thresholds (pixels): Min={min_cluster_size_threshold}, Max={max_cluster_size_threshold}")

    cluster_properties = [] # Danh sách lưu trữ thông tin các cụm hợp lệ
    for label_value in unique_labels_viz:
        if label_value == -1: continue # Bỏ qua điểm nhiễu

        cluster_mask = (clustered_label_img == label_value)
        cluster_size = np.sum(cluster_mask) # Số pixel trong cụm

        # --- Áp dụng bộ lọc kích thước ---
        if min_cluster_size_threshold < cluster_size < max_cluster_size_threshold:
            # Tính các đặc tính khác của cụm hợp lệ
            avg_intensity = np.mean(img_processed[cluster_mask]) # Cường độ trung bình (trên ảnh đã resize)

            # Tính tọa độ trung tâm (centroid)
            y_indices, x_indices = np.where(cluster_mask)
            centroid_y = np.mean(y_indices)
            centroid_x = np.mean(x_indices)

            prop = {
                'label': label_value,
                'size': cluster_size,
                'avg_intensity': avg_intensity,
                'centroid': (centroid_y, centroid_x), # (y, x)
                'color': cluster_color_map[label_value] # Lưu màu để tham khảo
            }
            cluster_properties.append(prop)
            # In thông tin cụm ứng viên (có thể bật để debug)
            # print(f"  -> Candidate Cluster {label_value}: Size={cluster_size}, AvgIntensity={avg_intensity:.2f}, Centroid=({centroid_y:.1f}, {centroid_x:.1f})")
        # else:
            # In thông tin cụm bị loại (có thể bật để debug)
            # print(f"  Cluster {label_value} rejected: Size={cluster_size} (Outside threshold)")


    # --- Logic Chọn Lọc Cụm Tốt Nhất (Ví dụ Heuristic) ---
    # Giả định 1: Khối u thường có cường độ sáng hơn (hoặc tối hơn) môi trường xung quanh.
    # Giả định 2: Khối u thường nằm gần trung tâm ảnh hơn (không phải luôn đúng).
    # => Chọn cụm có cường độ trung bình CAO NHẤT trong số các cụm hợp lệ về kích thước.
    # *** Bạn cần điều chỉnh logic này cho phù hợp với đặc điểm khối u trong dữ liệu của bạn ***
    if cluster_properties:
        print(f"Found {len(cluster_properties)} candidate clusters passing size filter.")
        # Sắp xếp các cụm ứng viên theo cường độ trung bình giảm dần (sáng nhất lên đầu)
        cluster_properties.sort(key=lambda p: p['avg_intensity'], reverse=True)

        # Chọn cụm đầu tiên (sáng nhất) sau khi sắp xếp
        best_cluster_label = cluster_properties[0]['label']
        selected_prop = cluster_properties[0]
        print(f"Selected Cluster (Heuristic: Brightest valid): Label={best_cluster_label}, Size={selected_prop['size']}, AvgIntensity={selected_prop['avg_intensity']:.2f}")

        # --- Ví dụ Logic Thay Thế (Comment lại nếu không dùng): Chọn cụm gần tâm nhất ---
        # center_y, center_x = h / 2.0, w / 2.0
        # min_dist_sq = float('inf')
        # best_cluster_label = -1
        # for prop in cluster_properties:
        #     cy, cx = prop['centroid']
        #     dist_sq = (cy - center_y)**2 + (cx - center_x)**2 # Bình phương khoảng cách là đủ để so sánh
        #     if dist_sq < min_dist_sq:
        #         min_dist_sq = dist_sq
        #         best_cluster_label = prop['label']
        #         selected_prop = prop
        # if best_cluster_label != -1:
        #      print(f"Selected Cluster (Heuristic: Closest to center): Label={best_cluster_label}, Size={selected_prop['size']}, Dist_sq={min_dist_sq:.1f}")
        # else:
        #      print("Could not select cluster based on center distance heuristic.")

    else:
        print("No suitable candidate clusters found based on size heuristics.")


    # Tạo mặt nạ nhị phân cho cụm được chọn
    if best_cluster_label != -1:
        potential_tumor_mask[clustered_label_img == best_cluster_label] = 255 # Gán giá trị trắng (255)


    # 9. Resize Results Back to Original Shape (Phóng to kết quả về kích thước ảnh gốc)
    print("Resizing results back to original shape...")
    # Sử dụng phép nội suy INTER_NEAREST để tránh tạo ra giá trị nhãn/màu không mong muốn khi phóng to
    # Phóng to ảnh hiển thị các cụm màu
    clustered_image_viz_final = cv2.resize(clustered_image_viz,
                                           (original_shape[1], original_shape[0]), # (width, height) cho cv2.resize
                                           interpolation=cv2.INTER_NEAREST)

    # Phóng to mặt nạ khối u tiềm năng
    potential_tumor_mask_final = cv2.resize(potential_tumor_mask,
                                            (original_shape[1], original_shape[0]),
                                            interpolation=cv2.INTER_NEAREST)

    # Tùy chọn: Áp dụng bộ lọc Gaussian Blur nhẹ cho mask sau khi resize để làm mịn cạnh răng cưa,
    # sau đó nhị phân hóa lại để đảm bảo chỉ có 0 và 255.
    # potential_tumor_mask_final = cv2.GaussianBlur(potential_tumor_mask_final, (3,3), 0)
    # _, potential_tumor_mask_final = cv2.threshold(potential_tumor_mask_final, 127, 255, cv2.THRESH_BINARY)


    processing_end_time = time.time()
    print(f"--> Time for Cluster Viz, Heuristic, Resize: {processing_end_time - dbscan_time:.2f} seconds")
    print("-" * 30)
    print(f"Total processing time: {processing_end_time - start_time:.2f} seconds")
    print("-" * 30)


    # Trả về:
    # - Ảnh gốc ban đầu (không thay đổi)
    # - Ảnh hiển thị các cụm màu (đã resize về kích thước gốc)
    # - Mặt nạ khối u tiềm năng (đã resize về kích thước gốc)
    # - Nhãn DBSCAN thô (của ảnh đã resize, không thay đổi kích thước)
    return img, clustered_image_viz_final, potential_tumor_mask_final, labels

# ===============================================
# --- Phần thực thi chính (Main Execution) ---
# ===============================================
if __name__ == "__main__":
    # !!! THAY ĐỔI ĐƯỜNG DẪN NÀY TỚI FILE ẢNH CỦA BẠN !!!
    # Ví dụ: 'C:/Users/Admin/Desktop/mri_scans/patient01_slice15.png'
    # Hoặc: '/home/user/data/brain_tumor.dcm'
    image_path = 'licensed-image.jpg' # <--- THAY ĐỔI Ở ĐÂY

    # --- CÁC THAM SỐ CẦN TINH CHỈNH ---

    # 1. Hệ số giảm kích thước (Downscale Factor)
    #    QUAN TRỌNG NHẤT cho ảnh lớn (như 2048x2048) để tránh treo máy.
    #    - 1.0: Kích thước gốc (Rất chậm/Có thể treo máy với ảnh lớn).
    #    - 0.5: Nửa kích thước (VD: 2048x2048 -> 1024x1024). Nhanh hơn đáng kể.
    #    - 0.25: 1/4 kích thước (VD: 2048x2048 -> 512x512). Thường là lựa chọn tốt để bắt đầu.
    #    - 0.125: 1/8 kích thước (VD: 2048x2048 -> 256x256). Rất nhanh nhưng có thể mất chi tiết.
    downscale_factor = 0.25 # <--- THỬ THAY ĐỔI GIÁ TRỊ NÀY TRƯỚC TIÊN

    # 2. Tham số DBSCAN (eps, min_samples)
    #    !!! PHỤ THUỘC RẤT NHIỀU VÀO 'downscale_factor' VÀ DỮ LIỆU CỤ THỂ !!!
    #    => Bạn PHẢI thử nghiệm và điều chỉnh các giá trị này sau khi chọn `downscale_factor`.
    #
    #    - `eps`: Ngưỡng khoảng cách trong không gian đặc trưng đã scale.
    #        + Tăng `eps`: Kết nối nhiều điểm hơn, tạo cụm lớn hơn, ít nhiễu hơn.
    #        + Giảm `eps`: Yêu cầu điểm gần nhau hơn, tạo cụm chặt hơn, nhiều nhiễu hơn.
    #        + Gợi ý khoảng thử nghiệm (SAU KHI ĐÃ SCALE): 0.5 đến 5.0 (hoặc hơn/kém tùy dữ liệu).
    #
    #    - `min_samples`: Số điểm tối thiểu trong vùng lân cận `eps` để tạo thành lõi cụm.
    #        + Tăng `min_samples`: Yêu cầu mật độ cao hơn, cụm nhỏ/thưa có thể thành nhiễu.
    #        + Giảm `min_samples`: Cho phép hình thành cụm với mật độ thấp hơn.
    #        + Gợi ý giá trị: Phụ thuộc vào kích thước cụm mong muốn và mật độ điểm sau khi resize.
    #                     Có thể thử từ 10, 25, 50, 100, 200,...
    #
    #    => Gợi ý giá trị khởi đầu cho downscale=0.25 (CẦN TINH CHỈNH!):
    dbscan_eps = 1.8         # <--- THỬ THAY ĐỔI
    dbscan_min_samples = 50  # <--- THỬ THAY ĐỔI

    print("=" * 40)
    print(f"Bắt đầu phân đoạn ảnh: {image_path}")
    print(f"Tham số sử dụng:")
    print(f"  - Downscale Factor: {downscale_factor}")
    print(f"  - DBSCAN eps: {dbscan_eps}")
    print(f"  - DBSCAN min_samples: {dbscan_min_samples}")
    print("=" * 40)

    # Gọi hàm phân đoạn
    results = segment_tumor_dbscan(image_path,
                                   eps=dbscan_eps,
                                   min_samples=dbscan_min_samples,
                                   downscale_factor=downscale_factor)

    # Kiểm tra và hiển thị kết quả
    if results:
        original_image, clustered_image, potential_tumor_mask, labels_raw = results

        print("\nHiển thị kết quả...")
        # --- Hiển thị kết quả bằng Matplotlib ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Tạo cửa sổ với 3 ô ảnh

        # Ô 1: Ảnh gốc
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title(f'Ảnh Gốc ({original_image.shape[0]}x{original_image.shape[1]})')
        axes[0].axis('off') # Ẩn trục tọa độ

        # Ô 2: Ảnh các cụm màu (đã resize về gốc)
        axes[1].imshow(clustered_image) # Ảnh cluster đã là ảnh màu BGR/RGB
        axes[1].set_title(f'Các Cụm DBSCAN (eps={dbscan_eps}, min={dbscan_min_samples})')
        axes[1].axis('off')

        # Ô 3: Ảnh gốc chồng với mặt nạ khối u tiềm năng
        axes[2].imshow(original_image, cmap='gray')
        # Hiển thị mask chồng lên với độ trong suốt (alpha) và bảng màu 'jet' hoặc 'cool'
        axes[2].imshow(potential_tumor_mask, cmap='jet', alpha=0.5) # alpha=0.5 -> trong suốt 50%
        axes[2].set_title('Mặt Nạ Khối U Tiềm Năng (Heuristic)')
        axes[2].axis('off')

        # Thêm tiêu đề chung cho cả cửa sổ
        fig.suptitle(f"Kết quả Phân đoạn DBSCAN (Downscale={downscale_factor})", fontsize=16, y=0.97)
        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Điều chỉnh layout để không bị chồng lấn
        plt.show() # Hiển thị cửa sổ đồ thị

        # --- Tùy chọn: Lưu kết quả ra file ---
        save_output = False # Đặt thành True nếu muốn lưu ảnh
        if save_output:
             output_folder = "dbscan_segmentation_output"
             import os
             if not os.path.exists(output_folder):
                 os.makedirs(output_folder)
             base_name = os.path.splitext(os.path.basename(image_path))[0]

             # Lưu ảnh cluster (chuyển từ RGB của matplotlib sang BGR của OpenCV)
             save_clustered_path = os.path.join(output_folder, f"{base_name}_clustered_ds{downscale_factor}_eps{dbscan_eps}_min{dbscan_min_samples}.png")
             cv2.imwrite(save_clustered_path, cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR))

             # Lưu ảnh mask
             save_mask_path = os.path.join(output_folder, f"{base_name}_mask_ds{downscale_factor}_eps{dbscan_eps}_min{dbscan_min_samples}.png")
             cv2.imwrite(save_mask_path, potential_tumor_mask)

             # Lưu ảnh gốc chồng mask
             overlay_image = cv2.addWeighted(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR), 0.7,
                                            cv2.cvtColor(potential_tumor_mask, cv2.COLOR_GRAY2BGR), 0.3, 0) # Tạo ảnh overlay
             save_overlay_path = os.path.join(output_folder, f"{base_name}_overlay_ds{downscale_factor}_eps{dbscan_eps}_min{dbscan_min_samples}.png")
             cv2.imwrite(save_overlay_path, overlay_image)


             print(f"\nKết quả đã được lưu vào thư mục: '{output_folder}'")

    else:
        print("\nQuá trình phân đoạn không thành công hoặc bị lỗi.")

    print("\nChương trình kết thúc.")