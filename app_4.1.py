import sys
import os
import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QMessageBox, QListWidget, QComboBox, QSizePolicy, QSplitter, QSlider
from PyQt5.QtCore import Qt

import numpy as np
from scipy.signal import hilbert
import pandas as pd 

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None
        self.roi = None
        self.new_roi = None
        self.is_selecting = False
        self.selected_bar_index = None
        self.raw_rf = None 

    def initUI(self):
        self.setWindowTitle('Image Processor')
        self.setGeometry(100, 100, 1600, 1000)  # Large window size

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a splitter to allow resizing between left and right panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel layout (List, Dropdown, Load Data Button)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.load_image)  # Connect to load_image
        left_layout.addWidget(self.file_list)

        self.file_selector = QComboBox()
        left_layout.addWidget(self.file_selector)

        load_data_button = QPushButton('Load Data')
        load_data_button.clicked.connect(self.load_data)
        left_layout.addWidget(load_data_button)

        splitter.addWidget(left_panel)

        # Right panel layout (Image, Plot, Buttons)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Slider for navigating frames
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)  # Will be set dynamically based on the number of frames
        self.frame_slider.setValue(0)
        self.value = 0 
        self.frame_slider.valueChanged.connect(self.update_frame)
        right_layout.addWidget(self.frame_slider)  # Add the slider first

        # Matplotlib canvas for image display (70% of the height)
        self.image_fig = Figure(figsize=(12, 6),)  # Adjust the size to be slightly smaller in height
        self.image_canvas = FigureCanvas(self.image_fig)
        right_layout.addWidget(self.image_canvas)

        # Second canvas for additional plots or overlays (70% of the height)
        self.second_fig = Figure(figsize=(12, 6))  # Adjust the size to be slightly smaller in height
        self.second_canvas = FigureCanvas(self.second_fig)
        right_layout.addWidget(self.second_canvas)

        # Matplotlib canvas for bar plot display (30% of the height)
        self.plot_fig = Figure(figsize=(12, 3))  # Ensure the bar plot has a visible height
        self.plot_canvas = FigureCanvas(self.plot_fig)
        right_layout.addWidget(self.plot_canvas)

        # Buttons
        button_layout = QHBoxLayout()
        buttons = ['Save Session', 'Reset']
        for button_text in buttons:
            button = QPushButton(button_text)
            button.clicked.connect(getattr(self, button_text.lower().replace(' ', '_')))
            button_layout.addWidget(button)

        right_layout.addLayout(button_layout)

        splitter.addWidget(right_panel)

        # Set the splitter as the central layout
        layout = QHBoxLayout(central_widget)
        layout.addWidget(splitter)

        # Connect mouse events for selecting ROI
        self.second_canvas.mpl_connect('button_press_event', self.start_roi_selection)
        self.second_canvas.mpl_connect('motion_notify_event', self.update_roi_selection)
        self.second_canvas.mpl_connect('button_release_event', self.end_roi_selection)

    def load_data(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder_path:
            all_files = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.dcm'):  # Filter for DICOM files
                        all_files.append(os.path.join(root, file))
            
            unique_files = sorted(set(os.path.basename(file) for file in all_files))
            self.file_selector.clear()
            self.file_selector.addItems(unique_files)

            def update_file_list():
                selected_file = self.file_selector.currentText()
                matching_files = [file for file in all_files if os.path.basename(file) == selected_file]
                self.file_list.clear()
                self.file_list.addItems(matching_files)
            
            self.file_selector.currentTextChanged.connect(update_file_list)
            update_file_list()  # Update the file list on initial load



    def load_image(self):
        selected_item = self.file_list.currentItem()
        self.value = 0 
        if selected_item:
            file_path = selected_item.text()
            self.file_path = file_path 
            print("filepath",file_path)
            try:
                # Load DICOM file
                self.loaded_dicom = get_signal_data(file_path)  # Assuming this function loads the DICOM data correctly
                self.num_frames, height,width,channels = self.loaded_dicom.shape
                self.frame_slider.setMaximum(self.num_frames - 1)
                self.display_image(0)
                # Handle multi-frame DICOM (4D array: frames, height, width, channels)
                
                # Extract the file prefix for envelope data
                base_filename = os.path.basename(file_path)
                file_prefix = base_filename.split('.')[0]  # Extract '0_0' from '0_0.dcm'

                # Construct the corresponding folder name
                folder_name = f"raw_{file_prefix}.tar_extracted"
                folder_path = os.path.join(os.path.dirname(file_path), folder_name)

                env_file_path = None

                if os.path.isdir(folder_path):
                    # Look for the envelope file (env)
                    possible_files = [
                        "C3_large_env.raw.rf.npy",
                        "L15_large_env.raw.rf.npy"
                    ]

                    for filename in possible_files:
                        file_to_find = os.path.join(folder_path, filename)
                        if os.path.exists(file_to_find):
                            env_file_path = file_to_find
                            print(f"Found envelope file: {file_to_find}")
                            break
                else:
                    print(f"Folder {folder_path} does not exist.")

                # Load ENvelope file 
                if env_file_path:
                    self.env_file_path = env_file_path
                    env_data = np.load(env_file_path)
                    print("envelop data before",env_data.shape)
                    # Apply Hilbert transform and other transformations to all frames
                    self.transformed_env_data = []
                    for i in range(env_data.shape[2]):  # Loop over frames
                        data_2d = env_data[:, :, i]
                        analytic_signal = hilbert(data_2d, axis=0)
                        amplitude_envelope = np.abs(analytic_signal)

                        # Apply logarithmic transformation
                        amplitude_envelope = 20 * np.log10(1 + amplitude_envelope)

                        # Rotate and flip the envelope
                        amplitude_envelope = np.rot90(amplitude_envelope)
                        amplitude_envelope = np.flipud(amplitude_envelope)

                        self.transformed_env_data.append(amplitude_envelope)

                    self.transformed_env_data = np.array(self.transformed_env_data)
                    print("transformed shape ",self.transformed_env_data.shape)
                    _,env_depth,env_signals = self.transformed_env_data.shape
                    self.display_env_image(0)
                
                # Load raw file 
                raw_file_path = env_file_path.replace("env","rf")
                print(f"Found raw file: {raw_file_path}")
                self.loaded_raw_rf = np.load(raw_file_path)
                self.loaded_raw_rf = np.transpose(self.loaded_raw_rf,axes=(2,0,1)) # ->frames, signals, depth
                self.raw_rf = self.loaded_raw_rf[0]
                print("loaded raw rf shape",self.loaded_raw_rf.shape)
                _,raw_signals,raw_depth = self.loaded_raw_rf.shape

                self.ratio_env_raw = raw_depth/env_depth

                if not  raw_signals==env_signals: 
                    QMessageBox.warning(self, "Error", f"Missmatch between signalsize of envelop file and raw RF ")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Unable to load the DICOM image. Error: {str(e)}")

    def display_image(self, frame_index):
        self.image_fig.clear()
        ax = self.image_fig.add_subplot(111)
        
        
        frame = self.loaded_dicom[frame_index]
        if frame.ndim == 3 and frame.shape[-1] == 3:  # Color image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Update self.image with the current frame
        
        ax.imshow(frame, cmap='gray', aspect='auto')  # Adjust aspect to 'auto' to fill the figure
        displayed_width = ax.get_window_extent().width
        displayed_height = ax.get_window_extent().height
        print("displayed size",displayed_height,displayed_width)
        ax.axis('off')
        self.image_canvas.draw()

    def display_env_image(self, frame_index):
        if hasattr(self, 'transformed_env_data') and self.transformed_env_data is not None:
            self.second_fig.clear()
            ax = self.second_fig.add_subplot(111)
            self.image = self.transformed_env_data[frame_index]
            ax.imshow(self.image, cmap='gray', aspect='auto')  # Adjust aspect to 'auto' to fill the figure
            
            # Re-apply ROI rectangle if it exists
            if self.roi:
                x, y, w, h = self.roi
                rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
            
            #ax.axis('off')
            self.second_canvas.draw()



    def display_image_with_roi(self):
        current_frame_index = self.frame_slider.value()  # Get the current frame index from the slider
        self.display_env_image(current_frame_index)  # Pass the current frame index to display_image

        # Reapply the ROI rectangle
        ax = self.second_fig.gca()
        if self.roi:
            x, y, w, h = self.roi
            rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            self.second_canvas.draw()
        #ax.axis('off')
        self.second_canvas.draw()

    

    def update_frame(self, value):
        # Update the B-mode DICOM image
        if self.loaded_dicom is not None:
            self.display_image(value)

        # Update the transformed envelope image
        if hasattr(self, 'transformed_env_data') and self.transformed_env_data is not None:
            self.display_env_image(value)

        # Reapply ROI if it exists
        if self.roi:
            self.display_image_with_roi()
            self.generate_bar_plot()
    
        #update raw rf 
        self.raw_rf = self.loaded_raw_rf[value]
        self.value = value 

    def start_roi_selection(self, event):
        if self.image is not None and event.inaxes:
            self.is_selecting = True
            self.roi_start = (int(event.xdata), int(event.ydata))
        
    def update_roi_selection(self, event):
        if self.is_selecting and event.inaxes:
            current_pos = (int(event.xdata), int(event.ydata))
            x0, y0 = self.roi_start
            x1, y1 = current_pos
            
            self.roi = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self.display_image_with_roi()
        
    def end_roi_selection(self, event):
        
        if self.is_selecting:
            self.is_selecting = False
            if self.roi:
                self.generate_bar_plot()

    

    def print_roi_coordinates(self):
        if self.roi:
            x, y, w, h = self.roi
            print(f"ROI Coordinates: x={x}, y={y}, width={w}, height={h}")
        else:
            print("No ROI selected.")

    def generate_bar_plot(self):
        if self.image is not None and self.roi:
            x, y, w, h = self.roi
            full_width_roi = self.image[y:y+h, :]  # TODO on env image! 

            if full_width_roi.ndim == 3:
                full_width_gray = cv2.cvtColor(full_width_roi, cv2.COLOR_RGB2GRAY)
            else:
                full_width_gray = full_width_roi

            barplot,transformed_roi,ontop_roi = tissue_similarity(full_width_gray,roi_coords=self.roi,raw_rf = self.raw_rf,ratio_env_raw=self.ratio_env_raw)  # TODO add raw RF image here 
            self.transformed_roi = transformed_roi
            self.ontop_roi = ontop_roi

            self.plot_fig.clear()
            ax = self.plot_fig.add_subplot(111)
            self.bars = ax.bar(range(self.image.shape[1]), barplot, picker=True)
            #ax.set_title('Average Pixel Intensity Across Image Width')
            #ax.set_xlabel('Column Index')
            #ax.set_ylabel('Average Intensity')

            self.highlight_rect = None  # Store the hover rectangle reference

            self.plot_canvas.draw()

            # Connect the hover event
            self.plot_canvas.mpl_connect('motion_notify_event', self.on_hover)
            self.plot_canvas.mpl_connect('pick_event', self.on_bar_click)

    def on_hover(self, event):
        if event.inaxes == self.plot_fig.gca():
            # Clear the previous rectangle if it exists
            if self.highlight_rect:
                self.highlight_rect.remove()
                self.highlight_rect = None

            for bar in self.bars:
                if bar.contains(event)[0]:  # Check if the mouse is over the bar
                    # Draw a rectangle around the hovered bar
                    x = bar.get_x()
                    y = bar.get_y()
                    width = bar.get_width()
                    height = bar.get_height()
                    self.highlight_rect = self.plot_fig.gca().add_patch(
                        plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='green', facecolor='none')
                    )
                    break

            self.plot_canvas.draw()

    def on_bar_click(self, event):
        if isinstance(event.artist, plt.Rectangle):
            for bar in event.canvas.figure.get_axes()[0].patches:
                bar.set_facecolor('blue')
            event.artist.set_facecolor('orange')
            self.selected_bar_index = int(event.artist.get_x())
            self.plot_canvas.draw()
            self.compute_new_region(self.selected_bar_index)

    def compute_new_region(self, selected_bar_index):
        if self.roi:
            _, y, w, h = self.roi
            print(selected_bar_index)
            new_x = selected_bar_index - w // 2
            new_x = max(0, min(new_x, self.image.shape[1] - w))
            self.new_roi = (new_x, y, w, h)
            self.highlight_new_region()

    def highlight_new_region(self):
        
        self.display_image_with_roi()
        ax = self.second_fig.gca()
        if self.new_roi:
            x, y, w, h = self.new_roi
            rect = plt.Rectangle((x, y), w, h, edgecolor='green', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            self.second_canvas.draw()
            print(self.roi)
            print(self.new_roi)
            print("highlighted second image")

            #ax.axis('off')
            self.second_canvas.draw()

    def save_session(self):
        
        uid = self.file_path.split("\\")[-3]

        handheld_type = self.env_file_path.split("\\")[-1].split("_")[0]
        frame = self.value

        self.ontop_roi 
        v1,v2,h1,h2 = self.transformed_roi

        depth = self.raw_rf.shape[1]    
        roi_new_coords = scale_roi_coords(self.new_roi,self.ratio_env_raw)
        transformed_new_roi = transform_coords(roi_new_coords,depth)

        v1_new,v2_new,h1_new,h2_new = transformed_new_roi

        
        store_df = pd.DataFrame({"uid":[uid],"handheld_type":[handheld_type],"frame":[frame],
         "v1":[v1],"v2":[v2],"h1":[h1],"h2":[h2],
         "v1_new":[v1_new],"v2_new":[v2_new],"h1_new":[h1_new],"h2_new":[h2_new]})
        

        


        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Session")
        print(store_df)
        print(os.path.join(save_dir,f"{uid}.csv"))
        store_df.to_csv( os.path.join(save_dir,f"{uid}.csv"),index= False)
        QMessageBox.information(self, "Save Session", f"Session saved successfully to {save_dir}")
        

    def save_image_with_roi(self, path):
        self.display_image_with_roi()
        self.image_fig.savefig(path)

    def reset(self):
        self.image_fig.clear()
        self.second_fig.clear()
        self.plot_fig.clear()
        self.image_canvas.draw()
        self.second_canvas.draw()
        self.plot_canvas.draw()
        self.file_list.clear()
        self.file_selector.clear()
        self.image = None
        self.roi = None
        self.new_roi = None
        self.selected_bar_index = None
        self.value = 0 
        print("Reset complete.")

def get_signal_data(dicom_file_path):
    # Read the DICOM file
    dicom = pydicom.dcmread(dicom_file_path)
    
    # Extract the pixel array (signal data)
    signal_data = dicom.pixel_array
    
    # If needed, normalize or process the signal data here
    # Example: Normalize to the range 0-255 for visualization
    normalized_signal_data = cv2.normalize(signal_data, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized_signal_data



######

def transform_coords(coords,max_y):
    """ Transform  from (x left, y top, widht , height  ) where x is signal , y is depth 
                to  (x left, y bottom , width , height )
                 to (v1,v2,h1,h2)
    """
    x,y,w,h = coords
    y_new = max_y-y #changing the point of view from top left to top bottom
    
    v1 = y_new-h    
    v2 = y_new
    h1 = x
    h2 = x+w

    return v1,v2,h1,h2
    
    
def path2roi(img,roi):
    v1,v2,h1,h2 = roi
    y_max,_ = np.shape(img)
    return v1,y_max,h1,h2

def scale_roi_coords(roi_coords,scale):
    """scale roi from env to raw"""
    x,y,w,h = roi_coords
    return x,int(y*scale),w,int(h*scale)

def tissue_similarity(full_width_gray,roi_coords,raw_rf,ratio_env_raw):
    raw_rf = np.rot90(raw_rf)
    raw_rf = np.flipud(raw_rf)
    depth,signals = raw_rf.shape 

    roi_coords = scale_roi_coords(roi_coords,ratio_env_raw)
    transformed_roi = transform_coords(roi_coords,depth)
    ontop_roi = path2roi(raw_rf,transformed_roi)


    vertical = True
    subrois = get_sub_rois(ontop_roi,n=2,o=0,vertical=vertical)
    kernel_type = "cross_correlation"#"conv" #"cross_correlation"#"row_wise_conv"

    x_list = []
    y_list = []
    title_list = []
    for idx,sub_roi in enumerate(subrois):
        x,y,title = apply_kernel(raw_rf,sub_roi,kernel_type=kernel_type)
        x_list.append(x)
        y_list.append(y)
        title_list.append(title+str(idx))

    
    y = np.asarray(y_list).mean(axis=0)

    #x,y,title = apply_kernel(raw_rf,ontop_roi,kernel_type="conv")

    #we have to do zero padding to y!
    y_padded = np.zeros((signals,))
    
    dif = abs(len(y)-len(y_padded))//2
    y_padded[dif:dif+len(y)] = y 

    print("new_scaled_roi",transformed_roi)
    print("new_scaled_ontoproi",ontop_roi)



    return y_padded,transformed_roi,ontop_roi
    


def minmax(list_pre,a,b):
    """Normalize list_pre from [min,max] to [a,b]"""
    A,B = np.min(list_pre),np.max(list_pre) 
    return np.array([(x-A)/(B-A)*(b-a)+a for x in list_pre])

def path2roi(img,roi):
    """Coordinates for the tissue on top of the ROI """
    v1,v2,h1,h2 = roi
    y_max,_ = np.shape(img)
    return v2,y_max,h1,h2



def extract_kernel(img,ontop_roi):
    """ extract the mask, notice that image is rotated!"""
    v1,v2,h1,h2 = ontop_roi
    y_max,_ = np.shape(img)
    return img[y_max-v2:y_max-v1,h1:h2] 

def get_sub_rois(roi,n,o,vertical=True):
    H = roi[1]-roi[0]
    B = roi[3]-roi[2]

    if vertical:
        b = (B+(n-1)*o)//n
        b_int = int(b)
        sub_rois = [(roi[0],roi[1],a_i := i*(b_int-o)+roi[2],a_i+b_int) for i in range(n)]
    else:
        b = (H+(n-1)*o)//n
        b_int = int(b)
        sub_rois = [(a_i := i*(b_int-o)+roi[0],a_i+b_int,roi[2],roi[3]) for i in range(n)]
    
    assert b-o>1, "overlap too big"
    assert B>n,"n too big "
    
    return sub_rois

def apply_kernel(img,ontop_roi,kernel_type="conv",getall=False):
    k_types = ["conv","row_wise_conv","cross_correlation"]
    if getall:
        #recursive call for all k_types
        x_all,y_all = [],[]
        for k_type in k_types:
            x,y,_ = apply_kernel(img,ontop_roi,kernel_type=k_type,getall=False)
            x_all.append(x)
            y_all.append(y)
        return x_all,y_all,k_types

    
    v1,v2,h1,h2 = ontop_roi
    width = h2-h1
    img = minmax(img.copy(),-1,1)
    y_max,x_max = np.shape(img)

    if kernel_type == "conv":
        flipped_kernel =  np.flip(extract_kernel(img,ontop_roi), axis=(0, 1)).copy()
    elif kernel_type == "row_wise_conv":
        flipped_kernel =  np.flip(extract_kernel(img,ontop_roi), axis=(1)).copy()
    elif kernel_type == "cross_correlation":
       flipped_kernel = extract_kernel(img,ontop_roi).copy()
    else:
        raise ValueError(f'kernel_type not known,only {k_types}')
    

    
    hist_list = []
    for i in range(x_max -width +1):
        area = img[y_max-v2:y_max-v1,i:i+width].copy()
        #area /= np.sum(area)
        hist_list.append(np.sum(area*flipped_kernel))

    normalized =minmax(hist_list,0,1)
    x = [i+width//2 for i in range(len(hist_list))]
    y = np.asarray(normalized)
    return x,y,kernel_type


def main():
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
