from moviepy.editor import VideoFileClip, concatenate_videoclips

def cut(address,start,end):
    clipOri = VideoFileClip(address)

    start = int(start) / 30
    end = int(end) / 30
    finalClip = clipOri.subclip(start,end)
    
    if address == './input/video_left.mp4':
        finalClip.write_videofile("./output/left_cut.mp4")
    if address == './input/video_right.mp4':
        finalClip.write_videofile("./output/right_cut.mp4")
