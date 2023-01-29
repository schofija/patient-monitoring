# Installation steps for person tracking system implemented with SORT algorithm and Yolov7

1. Clone yolov7 repo
    - https://github.com/WongKinYiu/yolov7.git
2. In a separate folder clone this branch of the patient monitoring repo
3. Copy the files from patient monitoring into the yolov7 directory
4. CD into the yolov7 directory
- (Optional) Create new virtual env here using the requirements.txt
    - I named mine yolov7_tracking
    - (anaconda) conda create --name <env_name> --file requirements.txt
- Or use pip install -r requirements.txt to add requirements to base environemnt
5. Run tracking with necessary tags
    - python detect_or_track.py --weights yolov7.pt --no-trace --view-img --nosave --source 0 --show-fps --track --classes 0 --show-track