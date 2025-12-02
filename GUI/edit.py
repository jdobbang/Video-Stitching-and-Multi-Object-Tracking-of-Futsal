try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy import VideoFileClip, concatenate_videoclips
import os
from tkinter import messagebox

def cut(address, start, end):
    try:
        clipOri = VideoFileClip(address)
        
        # Get video duration
        duration = clipOri.duration
        fps = clipOri.fps
        
        # Convert frame numbers to seconds
        start_sec = int(start) / fps
        end_sec = int(end) / fps
        
        # Validate time range
        if start_sec < 0:
            messagebox.showerror("Error", "Start frame cannot be negative!")
            clipOri.close()
            return
        
        if end_sec > duration:
            messagebox.showerror("Error", 
                f"End frame ({end}) exceeds video duration!\n"
                f"Video duration: {duration:.2f} seconds ({int(duration * fps)} frames at {fps:.2f} fps)\n"
                f"Maximum end frame: {int(duration * fps)}")
            clipOri.close()
            return
        
        if start_sec >= end_sec:
            messagebox.showerror("Error", "Start frame must be less than end frame!")
            clipOri.close()
            return
        
        print(f"Video info: {duration:.2f}s, {fps:.2f} fps")
        print(f"Cutting from {start_sec:.2f}s to {end_sec:.2f}s")
        
        # Try both subclip (moviepy 1.x) and subclipped (moviepy 2.x)
        try:
            finalClip = clipOri.subclip(start_sec, end_sec)
        except AttributeError:
            finalClip = clipOri.subclipped(start_sec, end_sec)
        
        # Create output directory if it doesn't exist
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename based on input filename
        basename = os.path.basename(address)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}_cut{ext}")
        
        print(f"Cutting video: {address}")
        print(f"Output will be saved to: {output_path}")
        finalClip.write_videofile(output_path)
        print(f"Video saved successfully to: {output_path}")
        
        messagebox.showinfo("Success", f"Video cut successfully!\nSaved to: {output_path}")
        
        clipOri.close()
        finalClip.close()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to cut video:\n{str(e)}")
        print(f"Error: {e}")
