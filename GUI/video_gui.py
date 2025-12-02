from tkinter import *
from tkinter import ttk, messagebox
from tkinter.filedialog import *
import os

#function files
import stitching as st
import video_trackbar as vt
import frames2Video as f2v
import edit
import stitching_coordinate as stc
# object_tracking will be imported when needed to avoid torch/YOLO import errors

#임시 function
def donothing():
   filewin = Toplevel(window)
   button = Button(filewin, text="Do nothing button")
   button.pack()

def check_files_selected():
    """Check if both video files are selected"""
    if 'left_video' not in filename_record or 'right_video' not in filename_record:
        messagebox.showerror("Error", "Please select both left and right video files first!")
        return False
    return True

def safe_edit():
    if check_files_selected():
        vt.trackbar(filename_record['left_video'], filename_record['right_video'])

def safe_cut_left():
    if 'left_video' not in filename_record:
        messagebox.showerror("Error", "Please select left video file first!")
        return
    edit.cut(filename_record['left_video'], e1.get(), e2.get())

def safe_cut_right():
    if 'right_video' not in filename_record:
        messagebox.showerror("Error", "Please select right video file first!")
        return
    edit.cut(filename_record['right_video'], e3.get(), e4.get())

def safe_generate_frames():
    if check_files_selected():
        # Use the cut videos from output folder
        left_basename = os.path.basename(filename_record['left_video'])
        right_basename = os.path.basename(filename_record['right_video'])
        
        left_name = os.path.splitext(left_basename)[0]
        right_name = os.path.splitext(right_basename)[0]
        left_ext = os.path.splitext(left_basename)[1]
        right_ext = os.path.splitext(right_basename)[1]
        
        left_cut = os.path.join("./output", f"{left_name}_cut{left_ext}")
        right_cut = os.path.join("./output", f"{right_name}_cut{right_ext}")
        
        # Check if cut videos exist
        if not os.path.exists(left_cut) or not os.path.exists(right_cut):
            messagebox.showerror("Error", 
                f"Cut videos not found!\n"
                f"Please cut both videos first.\n\n"
                f"Looking for:\n{left_cut}\n{right_cut}")
            return
        
        st.stitch(e5.get(), e6.get(), e7.get(), e8.get(), left_cut, right_cut)

def safe_track():
    # Import OpenCV-based object_tracking (no torch dependency)
    try:
        import object_tracking_opencv as ot
    except ImportError as e:
        messagebox.showerror("Error", 
            f"Cannot import object_tracking_opencv module:\n{str(e)}")
        return
    except Exception as e:
        messagebox.showerror("Error",
            f"Error loading tracking module:\n{str(e)}")
        return
    
    # Check if cylindricalFrames directory exists
    frames_dir = "./output/cylindricalFrames"
    
    if not os.path.exists(frames_dir):
        messagebox.showerror("Error",
            "Cylindrical frames not found!\n"
            "Please generate frames first.\n\n"
            f"Looking for: {frames_dir}")
        return
    
    # Check if directory has frames
    import glob
    frame_count = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if frame_count == 0:
        messagebox.showerror("Error",
            "No frames found in cylindricalFrames directory!\n"
            "Please generate frames first.")
        return
    
    messagebox.showinfo("Info", f"Found {frame_count} frames. Starting tracking...")
    
    # Get field corner coordinates
    try:
        field_corners = [
            [float(e9.get()), float(e10.get())],   # Top-Left
            [float(e11.get()), float(e12.get())],  # Top-Right
            [float(e13.get()), float(e14.get())],  # Bottom-Right
            [float(e15.get()), float(e16.get())]   # Bottom-Left
        ]
    except ValueError:
        messagebox.showerror("Error", "Invalid field coordinates! Please enter numeric values.")
        return
    
    # Create args object similar to argparse
    class Args:
        def __init__(self):
            self.model = 'yolov8n.onnx'  # Use ONNX model (will auto-download)
            self.conf = 0.5
            self.fps = 0
            self.src = frames_dir
            self.out_dir = './tracking_output'
            self.field_corners = field_corners
    
    # Create output directory
    if not os.path.exists('./tracking_output'):
        os.makedirs('./tracking_output')
    
    try:
        args = Args()
        ot.main(args)
        messagebox.showinfo("Success", "Tracking completed!\nResults saved to ./tracking_output/")
    except Exception as e:
        messagebox.showerror("Error", f"Tracking failed:\n{str(e)}")

#왼쪽 비디오 선택
def OpenFileL(file_record):
    file_record['left_video'] = askopenfilename(title="Select Left Video", filetypes=(("video files1","*.mp4"),("video files2","*.avi")))
    if file_record.get('left_video'):
        left_label.config(text=f"Left: {file_record['left_video'].split('/')[-1]}", fg="green")
    
#오른쪽 비디오 선택
def OpenFileR(file_record):
    file_record['right_video'] = askopenfilename(title="Select Right Video", filetypes=(("video files1","*.mp4"),("video files2","*.avi")))
    if file_record.get('right_video'):
        right_label.config(text=f"Right: {file_record['right_video'].split('/')[-1]}", fg="green")
    
#window 선언
window = Tk()
filename_record = {}

# 제목, 사이즈, 크기 조절
window.title("Video Stitching & Tracking - Futsal")
window.geometry("750x900")
window.resizable(True, True)

#메뉴
menubar = Menu(window)

videomenu = Menu(menubar, tearoff=0)
videomenu.add_command(label="Edit", command=safe_edit)
videomenu.add_command(label="Stitch Confirm", command=lambda: stc.stitch())
menubar.add_cascade(label="Tools", menu=videomenu)
window.config(menu=menubar)

# Create canvas with scrollbar for scrolling
canvas = Canvas(window)
scrollbar = Scrollbar(window, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Main container with padding
main_frame = Frame(scrollable_frame, padx=20, pady=20)
main_frame.pack(fill=BOTH, expand=True)

# === File Selection Section ===
file_frame = LabelFrame(main_frame, text="📁 Video Files", font=("Arial", 10, "bold"), padx=10, pady=10)
file_frame.pack(fill=X, pady=(0, 15))

# Left video selection
left_btn_frame = Frame(file_frame)
left_btn_frame.pack(fill=X, pady=(5, 5))
Button(left_btn_frame, text="📂 Select Left Video", bg="#3F51B5", fg="white", font=("Arial", 9, "bold"),
       command=lambda: OpenFileL(filename_record), pady=5).pack(side=LEFT, padx=(0, 10))
left_label = Label(left_btn_frame, text="Left: No file selected", anchor=W, fg="gray")
left_label.pack(side=LEFT, fill=X, expand=True)

# Right video selection
right_btn_frame = Frame(file_frame)
right_btn_frame.pack(fill=X, pady=(5, 5))
Button(right_btn_frame, text="📂 Select Right Video", bg="#3F51B5", fg="white", font=("Arial", 9, "bold"),
       command=lambda: OpenFileR(filename_record), pady=5).pack(side=LEFT, padx=(0, 10))
right_label = Label(right_btn_frame, text="Right: No file selected", anchor=W, fg="gray")
right_label.pack(side=LEFT, fill=X, expand=True)

# === Left Video Section ===
left_frame = LabelFrame(main_frame, text="⬅️ Left Video Settings", font=("Arial", 10, "bold"), padx=10, pady=10)
left_frame.pack(fill=X, pady=(0, 10))

Label(left_frame, text="Enter start and end frame:", font=("Arial", 9)).pack(anchor=W, pady=(5, 5))

frame_left = Frame(left_frame)
frame_left.pack(fill=X)
Label(frame_left, text="Start:", width=10, anchor=E).pack(side=LEFT, padx=(0, 5))
e1 = Entry(frame_left, width=15)
e1.pack(side=LEFT, padx=(0, 10))
e1.insert(0, "0")
Label(frame_left, text="End:", width=8, anchor=E).pack(side=LEFT, padx=(0, 5))
e2 = Entry(frame_left, width=15)
e2.pack(side=LEFT)
e2.insert(0, "1000")

btn2 = Button(left_frame, text="✂️ Cut Left Video", bg="#4CAF50", fg="white", font=("Arial", 9, "bold"), 
              command=safe_cut_left, pady=5)
btn2.pack(fill=X, pady=(10, 0))

# === Right Video Section ===
right_frame = LabelFrame(main_frame, text="➡️ Right Video Settings", font=("Arial", 10, "bold"), padx=10, pady=10)
right_frame.pack(fill=X, pady=(0, 10))

Label(right_frame, text="Enter start and end frame:", font=("Arial", 9)).pack(anchor=W, pady=(5, 5))

frame_right = Frame(right_frame)
frame_right.pack(fill=X)
Label(frame_right, text="Start:", width=10, anchor=E).pack(side=LEFT, padx=(0, 5))
e3 = Entry(frame_right, width=15)
e3.pack(side=LEFT, padx=(0, 10))
e3.insert(0, "0")
Label(frame_right, text="End:", width=8, anchor=E).pack(side=LEFT, padx=(0, 5))
e4 = Entry(frame_right, width=15)
e4.pack(side=LEFT)
e4.insert(0, "1000")

btn4 = Button(right_frame, text="✂️ Cut Right Video", bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
              command=safe_cut_right, pady=5)
btn4.pack(fill=X, pady=(10, 0))

# === Stitching Coordinates Section ===
coord_frame = LabelFrame(main_frame, text="📐 Stitching Coordinates", font=("Arial", 10, "bold"), padx=10, pady=10)
coord_frame.pack(fill=X, pady=(0, 10))

Label(coord_frame, text="Enter corner coordinates:", font=("Arial", 9)).pack(anchor=W, pady=(5, 5))

coord_grid = Frame(coord_frame)
coord_grid.pack(fill=X)

Label(coord_grid, text="Top-left X:", width=12, anchor=E).grid(row=0, column=0, padx=(0, 5), pady=3)
e5 = Entry(coord_grid, width=10)
e5.grid(row=0, column=1, padx=(0, 15), pady=3)
e5.insert(0, "0")

Label(coord_grid, text="Top-left Y:", width=12, anchor=E).grid(row=0, column=2, padx=(0, 5), pady=3)
e6 = Entry(coord_grid, width=10)
e6.grid(row=0, column=3, pady=3)
e6.insert(0, "0")

Label(coord_grid, text="Bottom-right X:", width=12, anchor=E).grid(row=1, column=0, padx=(0, 5), pady=3)
e7 = Entry(coord_grid, width=10)
e7.grid(row=1, column=1, padx=(0, 15), pady=3)
e7.insert(0, "1920")

Label(coord_grid, text="Bottom-right Y:", width=12, anchor=E).grid(row=1, column=2, padx=(0, 5), pady=3)
e8 = Entry(coord_grid, width=10)
e8.grid(row=1, column=3, pady=3)
e8.insert(0, "1080")

btn_generate = Button(coord_frame, text="🎬 Generate Frames", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
              command=safe_generate_frames, pady=10)
btn_generate.pack(fill=X, pady=(10, 0))

# === Field Coordinates Section ===
field_frame = LabelFrame(main_frame, text="⚽ Field Corner Coordinates (for tracking)", font=("Arial", 10, "bold"), padx=10, pady=10)
field_frame.pack(fill=X, pady=(0, 10))

Label(field_frame, text="Select 4 corners of the field (Top-Left → Top-Right → Bottom-Right → Bottom-Left):", font=("Arial", 9)).pack(anchor=W, pady=(5, 5))

field_grid = Frame(field_frame)
field_grid.pack(fill=X)

# Top-Left corner
Label(field_grid, text="Top-Left X:", width=12, anchor=E).grid(row=0, column=0, padx=(0, 5), pady=3)
e9 = Entry(field_grid, width=10)
e9.grid(row=0, column=1, padx=(0, 15), pady=3)
e9.insert(0, "100")

Label(field_grid, text="Top-Left Y:", width=12, anchor=E).grid(row=0, column=2, padx=(0, 5), pady=3)
e10 = Entry(field_grid, width=10)
e10.grid(row=0, column=3, pady=3)
e10.insert(0, "100")

# Top-Right corner
Label(field_grid, text="Top-Right X:", width=12, anchor=E).grid(row=1, column=0, padx=(0, 5), pady=3)
e11 = Entry(field_grid, width=10)
e11.grid(row=1, column=1, padx=(0, 15), pady=3)
e11.insert(0, "1820")

Label(field_grid, text="Top-Right Y:", width=12, anchor=E).grid(row=1, column=2, padx=(0, 5), pady=3)
e12 = Entry(field_grid, width=10)
e12.grid(row=1, column=3, pady=3)
e12.insert(0, "100")

# Bottom-Right corner
Label(field_grid, text="Bottom-Right X:", width=12, anchor=E).grid(row=2, column=0, padx=(0, 5), pady=3)
e13 = Entry(field_grid, width=10)
e13.grid(row=2, column=1, padx=(0, 15), pady=3)
e13.insert(0, "1820")

Label(field_grid, text="Bottom-Right Y:", width=12, anchor=E).grid(row=2, column=2, padx=(0, 5), pady=3)
e14 = Entry(field_grid, width=10)
e14.grid(row=2, column=3, pady=3)
e14.insert(0, "980")

# Bottom-Left corner
Label(field_grid, text="Bottom-Left X:", width=12, anchor=E).grid(row=3, column=0, padx=(0, 5), pady=3)
e15 = Entry(field_grid, width=10)
e15.grid(row=3, column=1, padx=(0, 15), pady=3)
e15.insert(0, "100")

Label(field_grid, text="Bottom-Left Y:", width=12, anchor=E).grid(row=3, column=2, padx=(0, 5), pady=3)
e16 = Entry(field_grid, width=10)
e16.grid(row=3, column=3, pady=3)
e16.insert(0, "980")

Label(field_frame, text="Real field size: 9m (width) × 16m (height)", font=("Arial", 8, "italic"), fg="gray").pack(anchor=W, pady=(5, 0))

btn_track = Button(field_frame, text="🎯 Track Objects", bg="#FF5722", fg="white", font=("Arial", 10, "bold"),
              command=safe_track, pady=10)
btn_track.pack(fill=X, pady=(10, 0))

# Bind mouse wheel for scrolling
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

window.mainloop()