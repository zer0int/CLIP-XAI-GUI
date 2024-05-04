import sys
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFileDialog, QSlider, QTextEdit)
from PySide2.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PySide2.QtCore import Qt, QRect
from PySide2.QtWidgets import QListWidget, QListWidgetItem
from PySide2.QtCore import Qt, QRect, Slot, QTimer, Signal
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox
import os
import subprocess
import argparse


class ClickableLabel(QLabel):
    roiUpdated = Signal(QRect)
    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.imageLoaded = False
        # Set a fixed size for the ROI box
        self.roi_box = QRect(0, 0, 20, 20)
        self.setMouseTracking(True)  # Enable mouse tracking to update the cursor position

    def mousePressEvent(self, event):
        # Update the top-left corner of the ROI box to where the user clicked
        self.roi_box.moveCenter(event.pos())
        self.update()
        self.roiUpdated.emit(self.roi_box)

    def paintEvent(self, event):
        super(ClickableLabel, self).paintEvent(event)
        if self.imageLoaded:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.drawRect(self.roi_box)


class ClipGui(QMainWindow):
    def __init__(self):
        super(ClipGui, self).__init__()
        self.ensure_directories_exist()
        self.setWindowTitle("CLIP GUI                                                                                      ")
        self.image_path = None
        self.selected_word = None
        self.saved_token_path = None
        self.saved_image_path = None
        self.tokens_file_last_modified = 0 
        self.last_mod_time = 0
        self.current_mod_time = 0
        self.model_choice = 'ViT-B/32'
        self.roi_box = QRect(0, 0, 20, 20)  # Initial position for the ROI box
        self.initUI()
      
    def ensure_directories_exist(self):
        required_dirs = ['clipapp', 'clipapp/tmp']
        for dir_path in required_dirs:
            # Check if the directory exists, and if not, create it
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
    
    def initUI(self):
        # Main widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Status Indicator initialization
        self.statusIndicator = QLabel("RDY")
        self.statusIndicator.setAlignment(Qt.AlignCenter)
        self.statusIndicator.setMinimumSize(60, 30)
        self.updateStatusIndicator("RDY")  # Set initial status

        # Feedback labels initialization
        self.feedback_model = QLabel("[3] CLIP opinion (select a word):")
        self.feedback_model.setAlignment(Qt.AlignCenter)
        self.feedback_model.setMinimumSize(60, 30)
        self.feedback_empty = QLabel("")
        self.feedback_empty.setAlignment(Qt.AlignCenter)
        self.feedback_empty.setMinimumSize(00, 00)

        # Adjust layout to include the status indicator and feedback labels
        feedback_layout = QHBoxLayout()
        feedback_layout.addWidget(self.feedback_model)
        feedback_layout.addWidget(self.feedback_empty)
        feedback_layout.addWidget(self.statusIndicator)  # Add the status indicator to the layout

        layout.addLayout(feedback_layout)  # Add feedback layout to the main QVBoxLayout
   
     
        # Inside initUI()
        self.words_list_widget = QListWidget()
        self.words_list_widget.itemClicked.connect(self.wordSelected)
        layout.addWidget(self.words_list_widget)
        self.setupListWidget()
        

        # Button to upload image
        self.upload_button = QPushButton("[1] Upload Image")
        self.upload_button.clicked.connect(self.uploadImage)
        layout.addWidget(self.upload_button)
      
     
        self.clip_opinion_button = QPushButton("[2] Get a CLIP opinion!")
        self.clip_opinion_button.clicked.connect(self.getClipOpinion)
        self.clip_opinion_button.setEnabled(True)
        layout.addWidget(self.clip_opinion_button)

        # New button for opening the tokens file
        self.open_tokens_button = QPushButton("📖📂")
        self.open_tokens_button.clicked.connect(self.openTokensFile)
        self.open_tokens_button.setEnabled(False)  # Disabled by default
        self.open_tokens_button.setStyleSheet("background-color: grey;")
        self.open_tokens_button.setToolTip("You can add your own words,\nseparated by a space and in a single line.")


        # Create a QHBoxLayout for these buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.clip_opinion_button, 90)  # Add with stretch factor of 80%
        buttons_layout.addWidget(self.open_tokens_button, 10)  # Add with stretch factor of 20%

        # Add the new layout to the main layout
        layout.addLayout(buttons_layout)
        
        self.empty_label = QLabel("")
        self.empty_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.empty_label)
       
        self.model_choice_combobox = QComboBox()
        model_options = [
            ('RN50 (5 GB)', 'RN50'),
            ('RN101 (7 GB)', 'RN101'),
            ('RN50x4 (11 GB)', 'RN50x4'),
            ('RN50x16 (27 GB) ⚠️', 'RN50x16'),
            ('RN50x64 (60 GB) ⚠️⚠️', 'RN50x64'),
            ('ViT-B/32 (4 GB)', 'ViT-B/32'),
            ('ViT-B/16 (9 GB)', 'ViT-B/16'),
            ('ViT-L/14 (22 GB)', 'ViT-L/14'),
            ('ViT-L/14@336px (43 GB) ⚠️', 'ViT-L/14@336px')
        ]
        for label, arg in model_options:
            self.model_choice_combobox.addItem(label, arg)  # Add the user-friendly label and associate it with the argument
        
        # Set 'ViT-B/32 (8 GB)' as the default by finding its index using the next line
        default_index = self.model_choice_combobox.findData('ViT-B/32')
        self.model_choice_combobox.setCurrentIndex(default_index)
        self.model_choice_combobox.currentIndexChanged.connect(self.updateModelChoice)
        layout.addWidget(self.model_choice_combobox)
        
        self.roi_info_label = QLabel("⚠️ Model size (VRAM) info:")
        self.roi_info_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.roi_info_label)
        info_layout = QHBoxLayout()

        # Add the model choice dropdown to the layout
        info_layout.addWidget(self.roi_info_label)  # Allocate 90% of the space

        # Create the information symbol label
        info_label = QLabel("🖱 Hover for Tooltip 📄")
        info_label.setToolTip("Model size (VRAM) only applies to 'Get a CLIP opinion!'\nYou can get a (small) CLIP's opinion,\nthen select a different model for use with [5]!")  # Set the tooltip text
        info_label.setAlignment(Qt.AlignCenter)  # Center align the text

        # Add the info label to the layout, allocating 10% of the space
        info_layout.addWidget(info_label)

        # Add the horizontal layout to the main layout
        layout.addLayout(info_layout)
        
        self.empty_label = QLabel("")
        self.empty_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.empty_label)

       
        self.roi_instruction_label = QLabel("[4] Click to select ROI:      ~ What did CLIP 👀 for (word) ?!")
        self.roi_instruction_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.roi_instruction_label)
        
        # Use the new ClickableLabel
        self.image_label = ClickableLabel()
        #self.image_label = QLabel("Click 'Upload Image' above.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(224, 224)
        layout.addWidget(self.image_label)
        self.image_label.roiUpdated.connect(self.onRoiUpdated)

        # Initialize the label for the heatmap
        self.heatmap_label = QLabel("Heatmap will be shown here.")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumSize(224, 224)

        # Create a horizontal layout for the images
        images_layout = QHBoxLayout()
        # Add the original image label and heatmap label to the images layout
        images_layout.addWidget(self.image_label)
        images_layout.addWidget(self.heatmap_label)
        # Add the images layout to the main vertical layout
        layout.addLayout(images_layout)

        self.confirm_button = QPushButton("[5] Confirm ROI / Get Heatmap")
        self.confirm_button.clicked.connect(self.onConfirmButtonClicked)
        layout.addWidget(self.confirm_button)
       
        # Feedback buttons
        self.feedback_positive = QLabel("Correct")
        self.feedback_positive.setAlignment(Qt.AlignCenter)
        self.feedback_negative = QLabel("Incorrect")
        self.feedback_negative.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.feedback_positive)
        layout.addWidget(self.feedback_negative)
    
    def onConfirmButtonClicked(self):
        vit_models = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        rn_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']

        if self.model_choice in vit_models:
            self.runClipexScript()
        elif self.model_choice in rn_models:
            self.runCliprnexScript()
        else:
            print("Unknown model selection.")

    def refreshWordListIfNeeded(self):
        img_name_without_ext = os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else ""
        tokens_file_path = f'clipapp/tokens_{img_name_without_ext}.txt'
        full_path = os.path.abspath(tokens_file_path)

        # Check if the file exists and has been modified
        if os.path.exists(full_path):
            current_mod_time = 0 #os.path.getmtime(full_path)
            if current_mod_time != self.last_mod_time:
                print("Tokens file has been updated. Refreshing list...")
                selected_texts = [item.text() for item in self.words_list_widget.selectedItems()]
                self.loadClipOutput(full_path)  # Reload contents into the list widget
                self.last_mod_time = current_mod_time
                # Re-select items based on stored text, if they still exist
                for index in range(self.words_list_widget.count()):
                    item = self.words_list_widget.item(index)
                    if item.text() in selected_texts:
                        item.setSelected(True)

    
    def updateModelChoice(self, index):
        model_options = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        self.model_choice = model_options[index]
        print(f"Model choice updated to: {self.model_choice}")
        self.heatmap_label.setText("Heatmap will be shown here.")
        self.heatmap_label.setStyleSheet("")  # Reset any custom styles if applied
        
    def updateStatusIndicator(self, status):
        """Update the status indicator's text and background color based on the status."""
        color_map = {
            "RDY": ("#90EE90", "black"),  # Light green background, black text
            "RUN": ("#FFD700", "black"),  # Yellow background, black text
            "FAIL": ("#FF6347", "white"),  # Tomato background, white text
        }
        color, text_color = color_map.get(status, ("grey", "black"))
        self.statusIndicator.setText(status)
        self.statusIndicator.setStyleSheet(f"background-color: {color}; color: {text_color};")

    # Update the uploadImage method to enable the button after an image is uploaded
    def uploadImage(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image_path = file_name
            self.updateImageDisplay()
            self.clip_opinion_button.setEnabled(True)  # Enable the button once an image is uploaded
            
            self.cropped_pixmap = QPixmap.fromImage(cropped_image)  # Store the cropped QPixmap globally
            self.image_label.setPixmap(self.cropped_pixmap.scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.saved_image_path = 'clipapp/tmp/saved_image.jpg'
            self.cropped_pixmap.save(self.saved_image_path)
            self.updateConfirmButtonState()
        else:
            self.updateConfirmButtonState()
            #QTimer.singleShot(3000, lambda: self.clip_opinion_button.setText("[2] Get a CLIP opinion!"))

    @Slot()
    def uploadImage(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image_path = file_name
            self.updateImageDisplay()

            # Load the image into a QImage object for easier manipulation
            image = QImage(file_name)
        
            # Determine the shortest side to calculate the square's dimensions
            size = min(image.width(), image.height())
        
            # Calculate the top-left corner of the square crop area
            left = (image.width() - size) // 2
            top = (image.height() - size) // 2

            # Crop the image to a square based on the shortest side
            cropped_image = image.copy(left, top, size, size)

            # Convert the QImage back to QPixmap to use the .save() method
            pixmap = QPixmap.fromImage(cropped_image)
            self.image_label.setPixmap(pixmap.scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # Save the image to a specific location for the script to access
            self.saved_image_path = 'clipapp/tmp/saved_image.jpg'
            pixmap.save(self.saved_image_path)
            self.updateConfirmButtonState()


    def updateImageDisplay(self):
        if self.image_path:
            original_pixmap = QPixmap(self.image_path).scaled(224, 224, Qt.KeepAspectRatio)
            # Create a temporary pixmap to draw the ROI box
            temp_pixmap = QPixmap(original_pixmap.size())
            temp_pixmap.fill(Qt.transparent)
            painter = QPainter(temp_pixmap)
            painter.drawPixmap(0, 0, original_pixmap) #(0, 0, original_pixmap)
            pen = QPen(QColor(255, 0, 0), 2)  # Red color for the ROI box
            painter.setPen(pen)
            painter.drawRect(self.roi_box)
            painter.end()
            self.image_label.setPixmap(temp_pixmap)
            
        if hasattr(self, 'cropped_pixmap'):
            temp_pixmap = self.cropped_pixmap.copy()  # Create a copy to draw the ROI
            painter = QPainter(temp_pixmap)
            pen = QPen(QColor(255, 0, 0), 2)  # Red color for the ROI box
            painter.setPen(pen)
            painter.drawRect(self.roi_box)
            painter.end()
            self.image_label.setPixmap(temp_pixmap.scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def onRoiUpdated(self, new_roi):
        self.roi_box = new_roi  # Update the main class' ROI box with the new one
        self.updateImageDisplay() 
        
    def confirmROI(self):
        self.updateConfirmButtonState()
            #QTimer.singleShot(3000, lambda: self.confirm_button.setText("[5] Confirm ROI Selection"))

           
    def setupListWidget(self):
        self.words_list_widget.itemSelectionChanged.connect(self.updateConfirmButtonState)
        
    def saveSelectedWord(self, word):
        self.saved_token_path = 'clipapp/tmp/saved_token.txt'
        with open(self.saved_token_path, 'w') as token_file:
            token_file.write(word)
            # If needed, find and select the item in the list here, by iterating through the list and matching the text.
        
    def restoreSelectionState(self):
        if self.selected_word:
            for index in range(self.words_list_widget.count()):
                item = self.words_list_widget.item(index)
                if item.text() == self.selected_word:
                    item.setSelected(True)
                return

    def updateConfirmButtonState(self):
        if not self.image_path:
            # No image uploaded
            self.confirm_button.setText("Please upload an image first!")
        elif not self.selected_word:
            # No word selected
            self.confirm_button.setText("Get a CLIP opinion + select a WORD first!")
        elif not self.loadClipOutput:
            # No word selected
            self.confirm_button.setText("Get a CLIP opinion + select a WORD first!")
        else:
            # Ready to confirm selection
            self.confirm_button.setText("[5] Confirm ROI / Get Heatmap")
        # Determine whether the button should be enabled
        self.confirm_button.setEnabled(bool(self.image_path and self.loadClipOutput and self.selected_word))

            
    def loadTokensToListWidget(self):
        """Loads tokens from the file to the list widget and updates the last modified timestamp."""
        tokens_file_path = self.getTokensFilePath()
        # Check if the file exists and get its last modified timestamp
        if os.path.exists(tokens_file_path):
            last_modified = 0 #os.path.getmtime(tokens_file_path)
            if last_modified != self.tokens_file_last_modified:
                print("Tokens file has been updated. Refreshing list...")
                self.tokens_file_last_modified = last_modified
                self.words_list_widget.clear()  # Clear existing items
                with open(tokens_file_path, 'r', encoding='utf-8') as file:
                    for word in file.read().strip().split():
                        self.words_list_widget.addItem(word)
                        self.updateConfirmButtonState()

    def getTokensFilePath(self):
        """Constructs and returns the tokens file path based on the current image name."""
        if self.image_path:
            img_name_without_ext = os.path.splitext(os.path.basename(self.image_path))[0]
            return f'clipapp/tokens_{img_name_without_ext}.txt'
        return ""

    @Slot(QListWidgetItem)
    def wordSelected(self, item):
        selected_text = item.text()  # Capture text at method start
        self.loadTokensToListWidget()  # Potentially modifies list
        self.refreshWordListIfNeeded()  # Potentially modifies list

        # Adjust selection logic
        if self.selected_word == selected_text:
            self.selected_word = None
        else:
            self.selected_word = selected_text
            self.saveSelectedWord(selected_text)
    
        self.restoreSelectionState()  # Restore selection based on self.selected_word
        self.updateConfirmButtonState()

                
    def compareROIWithBinaryMask(self, binary_mask_image_path):
        binary_mask_pixmap = QPixmap(binary_mask_image_path)
        mask_image = binary_mask_pixmap.toImage()

        # Count pixels in ROI
        matching_pixels = 0
        total_pixels = 0
        for x in range(self.roi_box.x(), self.roi_box.x() + self.roi_box.width()):
            for y in range(self.roi_box.y(), self.roi_box.y() + self.roi_box.height()):
                if mask_image.pixelColor(x, y) != QColor(0, 0, 0):  # Assuming non-black pixels mark attention areas
                    matching_pixels += 1
                total_pixels += 1

        # Determine if half or more of the ROI matches the binary mask
        if matching_pixels / total_pixels >= 0.5:
            self.feedback_positive.setStyleSheet("background-color: green")
            self.feedback_negative.setStyleSheet("")  # Reset other button
        else:
            self.feedback_negative.setStyleSheet("background-color: red")
            self.feedback_positive.setStyleSheet("")  # Reset other button
    
    
    def openTokensFile(self):
        img_name_without_ext = os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else ""
        tokens_file_path = f'clipapp/tokens_{img_name_without_ext}.txt'
        full_path = os.path.abspath(tokens_file_path)
        print("Full path to CLIP opinion:", full_path)

        if os.path.exists(full_path):
            # Update the file's timestamp by reloading and saving its content
            with open(full_path, 'r+', encoding='utf-8') as file:
                content = file.read()
                file.seek(0)
                file.write(content)
                file.truncate()
            self.last_mod_time = os.path.getmtime(full_path)  # Update the last_mod_time after the operation

            # Open the file for the user
            if os.name == 'nt':  # Windows
                os.startfile(full_path)
            elif os.name == 'posix':  # macOS, Linux
                try:
                    if os.uname().sysname == 'Darwin':
                        subprocess.run(['open', full_path])
                    else:
                        subprocess.run(['xdg-open', full_path])
                except AttributeError:
                    print("OS detection failed. Could not open file automatically.")
        else:
            print(f"File not found: {full_path}")



    
    def getClipOpinion(self):
        if self.image_path:
            # Reset heatmap label to its initial text
            self.heatmap_label.setText("Heatmap will be shown here.")
            self.heatmap_label.setStyleSheet("")  # Reset any custom styles if applied

            self.updateStatusIndicator("RUN")  # Update status to 'RUN'
            QApplication.processEvents() 
            img_name = os.path.basename(self.image_path)
            img_name_without_ext = os.path.splitext(img_name)[0]
            tokens_file_path = f'clipapp/tokens_{img_name_without_ext}.txt'
  
            self.clip_opinion_button.setText("[2] Get a CLIP opinion!")  # Reset the button text to default

            script_command = ['python', 'clipgaex-amp.py', self.image_path, self.model_choice]  # Ensure this is correct
            self.saved_token_path = tokens_file_path
            try:
                # Attempt to run the subprocess
                subprocess.run(script_command, check=True, stdout=None, stderr=subprocess.PIPE, text=True)
                print("\nSuccess. Loading CLIP opinion words...")
                self.loadClipOutput(tokens_file_path)
                self.updateStatusIndicator("RDY")  # Update status back to 'RDY' on success
                QApplication.processEvents()
                self.open_tokens_button.setEnabled(True)
                self.open_tokens_button.setStyleSheet("background-color: lightblue;")
                #self.last_mod_time = os.path.getmtime(tokens_file_path) 
                #self.last_mod_time = 0
                #self.current_mod_time = 0

                # Force-select the first item in the list, if present
                if self.words_list_widget.count() > 0:
                    self.words_list_widget.setCurrentRow(0)  # This automatically selects the first item
                    self.selected_word = self.words_list_widget.item(0).text()  # Update selected_word
                    self.saveSelectedWord(self.selected_word)  # Save the selected word
                    self.updateConfirmButtonState()  # Update the state of the confirm button
                    #self.last_mod_time = 0
                    #self.current_mod_time = 0
               
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while running CLIP gradient ascent: {e}")
                self.updateStatusIndicator("FAIL")  # Update status to 'FAIL' on error
                QApplication.processEvents()
        else:
            self.updateConfirmButtonState()

    def loadClipOutput(self, clip_output_path=''):
        self.words_list_widget.clear()  # Clear existing items
        try:
            with open(clip_output_path, 'r', encoding='utf-8') as file:
                clip_output = file.read().strip().split()
                for word in clip_output:
                    item = QListWidgetItem(word)
                    self.words_list_widget.addItem(item)
        except FileNotFoundError:
            self.text_output_edit.setPlainText("Could not load CLIP output, file not found.")

    def runClipexScript(self):
        # Check if the path for the tokens and a word have been set
        saved_image_path = None
        if self.saved_image_path and self.saved_token_path and self.selected_word:
            self.updateStatusIndicator("RUN")  # Set status to 'RUN' at the start
            QApplication.processEvents() 
            script_command = [
                'python', 'clipex.py',
                self.saved_image_path, 
                self.saved_token_path,
                self.model_choice
            ]
        
            print("\nObtaining Attention Heatmap...")
            try:
                # Run the script and capture output and errors
                completed_process = subprocess.run(script_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if completed_process.stderr:
                    print(f"Errors: {completed_process.stderr}")

                # Dynamically construct the heatmap image path based on the selected word
                heatmap_image_path = f'clipapp/saved_image_{self.selected_word}.png'
                binary_mask_image_path = f'clipapp/tmp/binary_mask_saved_image.png'

                # Load and display the heatmap image
                self.updateHeatmapDisplay(heatmap_image_path)

                # Check ROI against binary mask
                self.compareROIWithBinaryMask(binary_mask_image_path)
                self.updateStatusIndicator("RDY")  # Set status back to 'RDY' on success
                QApplication.processEvents()                 
            except subprocess.CalledProcessError:
                self.updateStatusIndicator("FAIL")  # Set status to 'FAIL' on error
                QApplication.processEvents() 
                
            else: 
                self.updateConfirmButtonState() 
            
            
    def runCliprnexScript(self):
        # Check if the path for the tokens and a word have been set
        if self.saved_image_path and self.saved_token_path and self.selected_word:
            self.updateStatusIndicator("RUN")  # Set status to 'RUN' at the start
            QApplication.processEvents() 
            script_command = [
                'python', 'cliprnex.py',
                self.saved_image_path, 
                self.saved_token_path,
                self.model_choice
            ]
        
            print("\nObtaining Attention Heatmap...")
            try:
                # Run the script and capture output and errors
                completed_process = subprocess.run(script_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if completed_process.stderr:
                    print(f"Errors: {completed_process.stderr}")

                # Dynamically construct the heatmap image path based on the selected word
                heatmap_image_path = f'clipapp/saved_image_{self.selected_word}.png'
                binary_mask_image_path = f'clipapp/tmp/binary_mask_saved_image.png'

                # Load and display the heatmap image
                self.updateHeatmapDisplay(heatmap_image_path)

                # Check ROI against binary mask
                self.compareROIWithBinaryMask(binary_mask_image_path)
                self.updateStatusIndicator("RDY")  # Set status back to 'RDY' on success
                QApplication.processEvents()                 
            except subprocess.CalledProcessError:
                self.updateStatusIndicator("FAIL")  # Set status to 'FAIL' on error
                QApplication.processEvents() 
                
        else:
            self.updateConfirmButtonState()
            
    def updateHeatmapDisplay(self, heatmap_image_path):
        # Check if the heatmap image file exists before attempting to load it
        if os.path.exists(heatmap_image_path):
            heatmap_pixmap = QPixmap(heatmap_image_path).scaled(224, 224, Qt.KeepAspectRatio)
            self.heatmap_label.setPixmap(heatmap_pixmap)
        else:
            print(f"Heatmap image not found: {heatmap_image_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { font-size: 12pt; }")

    mainWin = ClipGui()
    mainWin.show()
    sys.exit(app.exec_())
