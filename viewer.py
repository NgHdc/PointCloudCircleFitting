import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os
import threading
import time

# ==============================================================================
#                                  CẤU HÌNH
# ==============================================================================
# SỬA ĐƯỜNG DẪN đến file .txt của bạn
FILE_PATH = r"C:\Users\MAY02\Documents\E3C\NP-05_SCN0001.txt"

# THAM SỐ CHO RANSAC TỰ VIẾT
RANSAC_DISTANCE_THRESHOLD = 0.1  # Ngưỡng khoảng cách, ví dụ: 10cm
RANSAC_ITERATIONS = 2000         # Số lần lặp để tìm mô hình tốt
# ==============================================================================

# ------------------------------------------------------------------------------
# CÁC HÀM TOÁN HỌC CHO RANSAC TỰ VIẾT
# ------------------------------------------------------------------------------

def fit_circle_2d_from_3_points(p1, p2, p3):
    """Tính toán tâm và bán kính của đường tròn đi qua 3 điểm 2D."""
    D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    if abs(D) < 1e-8: # Các điểm gần như thẳng hàng
        return None, None

    p1_sq = p1[0]**2 + p1[1]**2
    p2_sq = p2[0]**2 + p2[1]**2
    p3_sq = p3[0]**2 + p3[1]**2

    center_x = (p1_sq * (p2[1] - p3[1]) + p2_sq * (p3[1] - p1[1]) + p3_sq * (p1[1] - p2[1])) / D
    center_y = (p1_sq * (p3[0] - p2[0]) + p2_sq * (p1[0] - p3[0]) + p3_sq * (p2[0] - p1[0])) / D
    
    center = np.array([center_x, center_y])
    radius = np.linalg.norm(p1 - center)
    
    return center, radius

def get_distances_to_cylinder(points_3d, model):
    """Tính khoảng cách hình học từ tất cả các điểm đến bề mặt hình trụ."""
    axis_dir, axis_point, radius = model
    
    # Sử dụng các phép toán vector hóa của NumPy để tăng tốc
    vectors_to_points = points_3d - axis_point
    dot_products = np.dot(vectors_to_points, axis_dir)
    projections_onto_axis = np.outer(dot_products, axis_dir)
    
    # dist_to_axis là khoảng cách trực giao từ mỗi điểm đến đường thẳng trục
    dist_to_axis = np.linalg.norm(vectors_to_points - projections_onto_axis, axis=1)
    
    # Khoảng cách đến bề mặt là |khoảng cách đến trục - bán kính|
    return np.abs(dist_to_axis - radius)

def run_custom_ransac(pcd, distance_threshold, iterations):
    """Thực thi thuật toán RANSAC tự viết để tìm hình trụ."""
    points = np.asarray(pcd.points)
    num_points = len(points)
    
    best_inlier_count = 0
    best_model = None
    best_inlier_indices = None

    for i in range(iterations):
        # 1. Chọn mẫu ngẫu nhiên gồm 3 điểm
        sample_indices = np.random.choice(num_points, 3, replace=False)
        sample_points = points[sample_indices]

        # 2. Tạo mô hình giả thuyết (Heuristic: hình trụ gần như thẳng đứng)
        p1_2d, p2_2d, p3_2d = sample_points[:, :2] # Chiếu 3 điểm xuống mặt phẳng XY
        center_2d, radius = fit_circle_2d_from_3_points(p1_2d, p2_2d, p3_2d)

        if center_2d is None or radius is None: # Mẫu bị suy biến (thẳng hàng)
            continue
            
        # Giả định trục thẳng đứng (0,0,1)
        axis_dir = np.array([0., 0., 1.])
        # Giả định điểm trên trục là tâm 2D nâng lên 3D với Z trung bình
        axis_point = np.array([center_2d[0], center_2d[1], np.mean(points[:, 2])])
        
        current_model = (axis_dir, axis_point, radius)

        # 3. Kiểm tra và đếm inliers
        distances = get_distances_to_cylinder(points, current_model)
        inlier_indices = np.where(distances < distance_threshold)[0]
        inlier_count = len(inlier_indices)
        
        # 4. Lưu kết quả tốt nhất
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = current_model
            best_inlier_indices = inlier_indices
            print(f"  RANSAC Iter {i+1}/{iterations}: Found new best model with {inlier_count} inliers.", end='\r')

    print("\n") # Xuống dòng sau khi vòng lặp kết thúc
    
    if best_inlier_indices is None:
        return None, None

    # 5. Tinh chỉnh cuối cùng (Refinement)
    final_inliers = points[best_inlier_indices]
    x_inliers, y_inliers = final_inliers[:, 0], final_inliers[:, 1]
    
    # Giải hệ phương trình A*x=b cho tâm và bán kính bằng bình phương tối thiểu
    A = np.c_[-2 * x_inliers, -2 * y_inliers, np.ones_like(x_inliers)]
    b = -(x_inliers**2 + y_inliers**2)
    try:
        solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c_term = solution
        refined_radius = np.sqrt(cx**2 + cy**2 - c_term)
        
        refined_axis_point = np.array([cx, cy, np.mean(final_inliers[:, 2])])
        refined_axis_dir = np.array([0., 0., 1.])
        
        # Gói kết quả vào một mảng 7 phần tử giống như output của Open3D
        refined_model_params = np.concatenate([refined_axis_dir, refined_axis_point, [refined_radius]])
        
        return refined_model_params, best_inlier_indices
    except np.linalg.LinAlgError:
        model_params = np.concatenate([best_model[0], best_model[1], [best_model[2]]])
        return model_params, best_inlier_indices

# ------------------------------------------------------------------------------
# LỚP ỨNG DỤNG GUI VÀ HÀM MAIN
# ------------------------------------------------------------------------------

def load_point_cloud(file_path):
    print(f"Attempting to load point cloud from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None
    try:
        points = np.loadtxt(file_path, delimiter=",", skiprows=1)
    except (ValueError, TypeError):
        try:
            print("Could not parse with comma delimiter, trying with space...")
            points = np.loadtxt(file_path, delimiter=" ", skiprows=1)
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    if points.shape[1] > 3:
        points = points[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

class AppWindow:
    def __init__(self, pcd):
        self.pcd_original = pcd
        self.points_original = np.asarray(pcd.points)
        self.pcd_filtered = o3d.geometry.PointCloud()
        self.pcd_filtered_for_fitting = None
        self.cylinder_model = None

        self.window = gui.Application.instance.create_window("Custom RANSAC Cylinder Fitter", 1280, 800)
        bbox = self.pcd_original.get_axis_aligned_bounding_box()
        self.data_z_min, self.data_z_max = bbox.get_min_bound()[2], bbox.get_max_bound()[2]
        self.z_min, self.z_max = self.data_z_min, self.data_z_max

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.1, 0.2, 0.3, 1.0])
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

        em = self.window.theme.font_size
        self.controls_panel = gui.Vert(0, gui.Margins(em, em, em, em))
        self._setup_controls()

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.controls_panel)
        self.window.set_on_layout(self._on_layout)
        self.update_filter()

    def _setup_controls(self):
        self.controls_panel.add_child(gui.Label("Z Min Threshold"))
        self.z_min_slider = gui.Slider(gui.Slider.DOUBLE)
        self.z_min_slider.set_limits(self.data_z_min, self.data_z_max)
        self.z_min_slider.double_value = self.z_min
        self.z_min_slider.set_on_value_changed(self._on_filter_changed)
        self.controls_panel.add_child(self.z_min_slider)

        self.controls_panel.add_child(gui.Label("Z Max Threshold"))
        self.z_max_slider = gui.Slider(gui.Slider.DOUBLE)
        self.z_max_slider.set_limits(self.data_z_min, self.data_z_max)
        self.z_max_slider.double_value = self.z_max
        self.z_max_slider.set_on_value_changed(self._on_filter_changed)
        self.controls_panel.add_child(self.z_max_slider)
        
        self.controls_panel.add_child(gui.Label("--------------------------"))
        self.fit_button = gui.Button("Fit Cylinder to Filtered Points")
        self.fit_button.set_on_clicked(self._on_fit_cylinder)
        self.controls_panel.add_child(self.fit_button)

        self.results_label = gui.Label("Fitting Results:\n- Not fitted yet")
        self.controls_panel.add_child(self.results_label)
        self.point_count_label = gui.Label("Filtered Points: 0")
        self.controls_panel.add_child(self.point_count_label)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene_widget.frame = r
        width = 20 * layout_context.theme.font_size
        height = min(r.height, self.controls_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self.controls_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_filter_changed(self, new_value):
        self.z_min, self.z_max = self.z_min_slider.double_value, self.z_max_slider.double_value
        if self.z_min > self.z_max:
            self.z_min = self.z_max
            self.z_min_slider.double_value = self.z_min
        threading.Thread(target=self.update_filter).start()

    def update_filter(self):
        z_mask = (self.points_original[:, 2] >= self.z_min) & (self.points_original[:, 2] <= self.z_max)
        filtered_points = self.points_original[z_mask]
        
        self.pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
        self.pcd_filtered.paint_uniform_color([0.7, 0.7, 0.7])
        self.pcd_filtered_for_fitting = o3d.geometry.PointCloud(self.pcd_filtered)
        text_to_show = f"Filtered Points: {len(filtered_points)}"
        
        def update_gui():
            self.scene_widget.scene.clear_geometry()
            if len(filtered_points) > 0:
                self.scene_widget.scene.add_geometry("filtered_pcd", self.pcd_filtered, rendering.MaterialRecord())
            self.point_count_label.text, self.results_label.text = text_to_show, "Fitting Results:\n- Ready to fit"
            
        gui.Application.instance.post_to_main_thread(self.window, update_gui)

    def _on_fit_cylinder(self):
        if not self.pcd_filtered_for_fitting or not self.pcd_filtered_for_fitting.has_points():
            return
        threading.Thread(target=self.run_ransac).start()

    def run_ransac(self):
        def update_status(text):
            self.results_label.text = f"Fitting Results:\n- {text}"
        gui.Application.instance.post_to_main_thread(self.window, lambda: update_status("Running Custom RANSAC..."))
        
        start_time = time.time()
        try:
            self.cylinder_model, inlier_indices = run_custom_ransac(
                self.pcd_filtered_for_fitting, RANSAC_DISTANCE_THRESHOLD, RANSAC_ITERATIONS
            )
            end_time = time.time()
            print(f"Custom RANSAC finished in {end_time - start_time:.2f} seconds.")
            if self.cylinder_model is None:
                 raise ValueError("Custom RANSAC failed to find a model.")
            self.update_fitting_results_gui(inlier_indices)
        except Exception as e:
            gui.Application.instance.post_to_main_thread(self.window, lambda: update_status(f"Error: {e}"))

    def update_fitting_results_gui(self, inlier_indices):
        def update_gui_and_scene():
            axis_dir = self.cylinder_model[:3]
            axis_point = self.cylinder_model[3:6]
            radius = self.cylinder_model[6]
            
            result_text = (f"Fitting Results:\n- Radius: {radius:.4f}\n- Axis Dir: [{axis_dir[0]:.2f}, {axis_dir[1]:.2f}, {axis_dir[2]:.2f}]\n- Num Inliers: {len(inlier_indices)}")
            self.results_label.text = result_text

            pcd_inliers = self.pcd_filtered_for_fitting.select_by_index(inlier_indices)
            pcd_inliers.paint_uniform_color([0.0, 0.8, 0.0])
            pcd_outliers = self.pcd_filtered_for_fitting.select_by_index(inlier_indices, invert=True)
            pcd_outliers.paint_uniform_color([0.8, 0.0, 0.0])
            
            self.scene_widget.scene.clear_geometry()
            self.scene_widget.scene.add_geometry("inliers", pcd_inliers, rendering.MaterialRecord())
            self.scene_widget.scene.add_geometry("outliers", pcd_outliers, rendering.MaterialRecord())
            
        gui.Application.instance.post_to_main_thread(self.window, update_gui_and_scene)

def main():
    pcd = load_point_cloud(FILE_PATH)
    if not pcd or not pcd.has_points():
        print("Could not load point cloud, exiting.")
        return

    gui.Application.instance.initialize()
    app = AppWindow(pcd)
    gui.Application.instance.run()

if __name__ == "__main__":
    main()