from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path
import glob
import os
from natsort import natsorted
from PIL import Image
import numpy as np

policy_folder_path = Path("./wandb/run-20250130_124812-1hlbzqhf/files/media/videos/train")
track_transformer_folder_path = Path("./wandb/0313_track_transformer/train")

cotracker_folder_path = Path("./data/preprocessed_demos/aloha_lamp/lamp_right_arm/episode_15/videos")

video_output_path = Path("./output_videos_new")
image_output_path = Path("./output_videos_new/images")



def merge_policy_or_track_transformer_videos(folder_path):
    # List of input video files (each is 0 seconds long, but you want the frame from each)


    video_files = glob.glob(os.path.join(folder_path, '*.mp4'))
    video_files = natsorted(video_files)

    clips = []
    # Loop through each video file and extract the first frame (since each video is 0 seconds long)
    for i in range(0, len(video_files), 50):
        clip = VideoFileClip(video_files[i])
        # Take only the first frame (use `subclip` to extract just the first second)
        # frame = clip.subclip(0, 1)
        clip = clip.set_duration(1.5)
        clips.append(clip)

    # Concatenate the clips to form the final video
    final_clip = concatenate_videoclips(clips, method="chain")

    # Set the duration of each frame to 1 second
    # final_clip = final_clip.set_duration(len(video_files))

    if folder_path == policy_folder_path:
        output_path = os.path.join(video_output_path, 'policy_video_merged.mp4')
    else:
        output_path = os.path.join(video_output_path, '0313_track_transformer_video_merged_50.mp4')

    # Write the result to a new video file
    final_clip.write_videofile(output_path, fps=25)

def merge_cotracker_videos(folder_path):
    # take every 2nd video
    # Get a list of all video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4'))]
    
    # List to store selected video clips
    video_clips = []
    
    # Iterate through the video files and select every second video
    for i in range(0, len(video_files), 2):
        video_file_path = os.path.join(folder_path, video_files[i])
        
        # Load the video clip
        clip = VideoFileClip(video_file_path)
        
        # Append to the list
        video_clips.append(clip)
    
    # Concatenate the selected video clips
    final_clip = concatenate_videoclips(video_clips)

    output_path = os.path.join(video_output_path, 'cotracker_video_merged.mp4')
    
    # Write the result to the output file
    final_clip.write_videofile(output_path)

def save_first_frame_as_image(folder_path):
    video_files = glob.glob(os.path.join(folder_path, '*.mp4'))
    video_files = natsorted(video_files)
    # video_file_path = os.path.join(folder_path, video_files[0])

    # Load the video
    clip = VideoFileClip(video_files[100])
    
    # Get the first frame (at time 0)
    first_frame = clip.get_frame(0)  # time=0 for the first frame
    
    # Convert the frame to an image (using PIL)
    first_frame_image = Image.fromarray(np.array(first_frame))
    
    # Save the image
    filename = "reconstruct_track_100.jpg"
    first_frame_image.save(os.path.join(image_output_path, filename))

def main():
    # merge_cotracker_videos(cotracker_folder_path)
    # merge_policy_or_track_transformer_videos(track_transformer_folder_path)
    # merge_policy_or_track_transformer_videos(policy_folder_path)
    save_first_frame_as_image(track_transformer_folder_path)

if __name__ == "__main__":
    main()