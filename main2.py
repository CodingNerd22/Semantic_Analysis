from moviepy.video.io.VideoFileClip import VideoFileClip
from utils import process_pdf, find_most_similar, find_full_context
import re

def extract_video_segment(input_video, output_video, start_time, end_time):
    """
    Extracts a segment from a video based on start and end timestamps.
    """
    # Load the video file
    video = VideoFileClip(input_video)
    
    # Extract the segment from the video
    segment = video.subclipped(start_time, end_time)
    
    # Save the extracted segment to a new file with audio
    segment.write_videofile(output_video, codec="libx264", audio=True, audio_codec="aac")
    
    # Close the video file
    video.close()

def parse_timestamps(text):
    """
    Parses the text output to extract the first and last timestamps.
    Returns the start and end times in seconds.
    """
    # Regex to match timestamps in the format "00:00:06,000 --> 00:00:12,074"
    timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")
    
    # Find all timestamps in the text
    matches = timestamp_pattern.findall(text)
    
    if not matches:
        raise ValueError("No timestamps found in the text output.")
    
    # Extract the first and last timestamps
    first_timestamp = matches[0][0]  # Start time of the first segment
    last_timestamp = matches[-1][1]  # End time of the last segment
    
    # Convert timestamps to seconds
    def timestamp_to_seconds(timestamp):
        hh, mm, ss_ms = timestamp.split(":")
        ss, ms = ss_ms.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000
    
    start_time = timestamp_to_seconds(first_timestamp)
    end_time = timestamp_to_seconds(last_timestamp)
    
    return start_time, end_time

def main():
    # Load PDF once
    pdf_path = input("Give name of the PDF: ")
    chunks, embeddings = process_pdf(pdf_path)
    
    # Interactive search loop
    while True:
        user_prompt = input("prompt: ")
        if user_prompt.lower() == 'exit':
            break
        
        # Find matches
        results = find_most_similar(user_prompt, chunks, embeddings, top_k=1)
        
        # Display results with context
        print(f"\nTop {len(results)} matches:")
        output_text = ""
        for idx, (chunk, score) in enumerate(results, 1):
            context = find_full_context(chunk, chunks)
            output_text += f"\nMatch #{idx} (Page {chunk['page']})\n"
            output_text += f"Context: {context}\n"
            output_text += "-" * 80 + "\n"
        
        print(output_text)
        
        # Parse timestamps from the output text
        try:
            start_time, end_time = parse_timestamps(output_text)
            print(f"Extracted timestamps: Start = {start_time}s, End = {end_time}s")
            
            # Extract video segment
            input_video = "/Users/user/VIIT/Backup/Semantic_analysis_test_2/Singham_video.mp4"
            output_video = "/Users/user/VIIT/Backup/Semantic_analysis_test_2/output_segment6.mp4"
            extract_video_segment(input_video, output_video, start_time, end_time)
            print(f"Video segment saved to {output_video}")
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()