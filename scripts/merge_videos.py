from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path
import glob
import os

policy_folder_path = Path("./wandb/run-20250130_124812-1hlbzqhf/files/media/videos/train")
track_transformer_folder_path = Path("./wandb/run-20250303_020028-5do80myc/files/media/videos/train")

cotracker_folder_path = Path("./data/preprocessed_demos/aloha_lamp/lamp_right_arm/episode_15/videos")

video_output_path = Path("./output_videos")



def merge_policy_or_track_transformer_videos(folder_path):
    # List of input video files (each is 0 seconds long, but you want the frame from each)


    video_files = glob.glob(os.path.join(folder_path, '*.mp4'))

    clips = []
    # Loop through each video file and extract the first frame (since each video is 0 seconds long)
    for i in range(len(video_files)):
        clip = VideoFileClip(video_files[i])
        # Take only the first frame (use `subclip` to extract just the first second)
        frame = clip.subclip(0, 1)
        clips.append(frame)

    # Concatenate the clips to form the final video
    final_clip = concatenate_videoclips(clips, method="chain")

    # Set the duration of each frame to 1 second
    final_clip = final_clip.set_duration(len(video_files))

    if folder_path == policy_folder_path:
        output_path = os.path.join(video_output_path, 'policy_video_merged.mp4')
    else:
        output_path = os.path.join(video_output_path, 'track_transformer_video_merged.mp4')

    # Write the result to a new video file
    final_clip.write_videofile(output_path, fps=24)

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

def main():
    # merge_cotracker_videos(cotracker_folder_path)
    merge_policy_or_track_transformer_videos(track_transformer_folder_path)
    # merge_policy_or_track_transformer_videos(policy_folder_path)

if __name__ == "__main__":
    main()