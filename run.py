from preprocessing import *


iemocap = IemocapData()
iemocap.divide_videos_to_clips()
iemocap.extract_video_frames(29)

