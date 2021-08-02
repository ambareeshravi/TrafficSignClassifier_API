netstat -ln | grep 8889

curl -X POST -F image=@/home/ambareesh/00008.png 'http://localhost:12345/predict'