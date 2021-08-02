# Traffic Sign Classification API
Training a CNN classifier model in tensorflow/keras to classify the type of traffic signal image and using an API online to upload image and get prediction
   
1. Download the German traffic sign dataset from https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

2. Install all the dependencies from requirements.text in your python3 environment
    
    ```python
    pip3 install -r requirements.txt
    ```
                
3. To train the Classification CNN model to classify traffic signs, run
    
    ```python
    python3 train.py
    ```
    
    The trained model will be saved in the following structure
    
    ```bash
    models/
    ├── sign_model.h5
    ...
    ```
    
4. To test, evaluate the trained model, run

    ```python
    python3 prediction.py
    ```
        
5. Files like data.py, utils.py , models.py support the execution of the project

6. Enable port execution by using
    ```bash
    netstat -ln | grep <port>
    ```

7. Test the API by using
    ```bash
    curl -X POST -F image=@/home/ambareesh/00008.png 'http://localhost:12345/predict'
    ```
