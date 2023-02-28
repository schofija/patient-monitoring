import cv2
import os

#script to take folder of sequential images and convert them to video at 30fps

def generate_video(video_name, image_folder, video_folder):
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
     
    # Array images should only consider
    # the image files ignoring others if any
  
    frame = cv2.imread(os.path.join(image_folder, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape 
    path = video_folder + '/' + video_name 
  
    video = cv2.VideoWriter(path, 0, 30, (width, height))
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated
    #os.chdir('..')
  
  
if __name__ == "__main__":

    # Calling the generate_video function
    cwd = os.getcwd() + "\\data\\"
    subdirs = [o for o in os.listdir(cwd) if os.path.isdir(os.path.join(cwd,o))]
    out_path = subdirs.pop(-1)
    os.chdir(cwd)
    for idx, x in enumerate(subdirs):
        name = "video_" + str(idx) + ".avi"
        wd = cwd + x
        generate_video(name, x, out_path)