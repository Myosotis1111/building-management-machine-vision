# building-management-machine-vision

The python project （see folder “detection_platform”） is packed using conda. Users can download the folder (please make sure there is at least 8GB available in the disk Anaconda installed), import to IDE and follow three steps to configure the environment:
(1) Use Conda to create a new environment. Run the following command in a terminal or command prompt:

conda env create -f environment.yaml

This will create a new Conda environment using the configuration in the environment.yaml file.
(2) After the environment is created successfully, activate the newly created environment. Run the following command:

conda activate rs_detection

(3) Finally, change the Python interpreter of the program to the interpreter in the conda environment you just created. The project can then be run via detection_platform.py.
