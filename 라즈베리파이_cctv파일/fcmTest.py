from pyfcm import FCMNotification

APIKEY = "AAAAybUThak:APA91bF-h9BdW07E1RM1GCUw7RrGC367viE7N7Z4DekyQV8serT_23Kh3KIL4qa0ojOFYTznUTgX8ifh41r9L_TF_V4CftiDZpa9D_R_lkyOacp5uJbbHF79vgGbSIP_j1lriTFylTJf"
TOKEN = "csvmBwBxRWubQ-HcdmKMya:APA91bEHRyPkzOyCS8EIsEusYcUtLR8kdLF12tD1CK_p0q2YlWCrCA4ssaMfe94OxyDcHXUWyTyBrVzSoe0E0ceiSHfXrrLeu5RNvPETVUNOW0fiFel0TRcDF2BbFd5IwAp1Dm349Olk"
 
# 파이어베이스 콘솔에서 얻어 온 서버 키를 넣어 줌
push_service = FCMNotification(APIKEY)
 
def sendMessage(title):
    # 메시지 (data 타입)

    file_name = "cctv_20220523_015349"
    image_type = 'image/png'
    file_token = "e3bc8f5d-240e-404e-a035-15d748f04a74"

    new_body = "https://firebasestorage.googleapis.com/v0/b/test-fffec.appspot.com/o/image_storage%2F" + file_name + "." + image_type + "?alt=media&token=" + file_token
    data_message = {
        "body": new_body,
        "title": title
    }
 
    # 토큰값을 이용해 1명에게 푸시알림을 전송함
    result = push_service.single_device_data_message(registration_id=TOKEN, data_message=data_message)
 
    # 전송 결과 출력
    print(result)
 
sendMessage("cctv test")