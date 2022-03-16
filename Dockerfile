FROM osrf/ros:noetic-desktop-full

# Setting Catkin Workspace
RUN /bin/bash -c "apt-get update &&\
    source /opt/ros/noetic/setup.bash &&\
    mkdir -p ~/catkin_ws/src &&\
    cd ~/catkin_ws/src &&\
    catkin_init_workspace &&\
    cd ~/catkin_ws &&\
    catkin_make &&\
    source devel/setup.bash"

# Install git
RUN /bin/bash -c "apt-get install -y git"

# Install usb-cam package
RUN /bin/bash -c "apt-get install -y ros-noetic-usb-cam"

# Install Python3 pip
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash &&\
    apt-get install -y python3-pip"

# Install Python Package for ocr
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash &&\
    apt-get install -y tesseract-ocr tesseract-ocr-script-hang tesseract-ocr-script-hang-vert &&\
    pip3 install pytesseract &&\
    pip3 install imutils"

# Install Python Package for Documentation
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash &&\
    pip3 install Sphinx &&\
    pip3 install sphinx_rtd_theme &&\
    apt-get install -y ros-noetic-rosdoc-lite"
