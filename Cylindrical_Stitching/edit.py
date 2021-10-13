from moviepy.editor import VideoFileClip, concatenate_videoclips

def cut(address,start,end):
    clipOri = VideoFileClip(address)

    start = int(start) / 30
    end = int(end) / 30
    finalClip = clipOri.subclip(start,end)
    
    if address == 'C:/Users/ehgus/Desktop/github/Video_Stitching_and_Multi_Object_Tracking_of_Futsal/Cylindrical_Stitching/input/video_left.mp4':
        finalClip.write_videofile("C:/Users/ehgus/Desktop/github/Video_Stitching_and_Multi_Object_Tracking_of_Futsal/Cylindrical_Stitching/output/left_cut.mp4")
    if address == 'C:/Users/ehgus/Desktop/github/Video_Stitching_and_Multi_Object_Tracking_of_Futsal/Cylindrical_Stitching/input/video_right.mp4':
        finalClip.write_videofile("C:/Users/ehgus/Desktop/github/Video_Stitching_and_Multi_Object_Tracking_of_Futsal/Cylindrical_Stitching/output/right_cut.mp4")
