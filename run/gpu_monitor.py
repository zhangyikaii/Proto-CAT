import smtplib
from email.mime.text import MIMEText
from email.header import Header

import time
import sys
sys.path.append("../")
from models.utils import gpu_state

polling_interval = {'success': 120, 'fail': 60} # 单位: 秒.

def send_mail(sendTo='zykycy@gmail.com'):
    sendFrom = '645064582@qq.com'
    smtp_server = 'smtp.qq.com'
    msg = MIMEText('有了宝!', 'plain', 'utf-8')

    msg['From'] = Header(sendFrom)
    msg['To'] = Header(sendTo)
    msg['Subject'] = Header('可')

    server = smtplib.SMTP_SSL(host=smtp_server)
    server.connect(smtp_server, 465)
    server.login(sendFrom, '')
    server.sendmail(sendFrom, sendTo, msg.as_string())
    server.quit()

gpus = '0,1'
gpu_available = ''
space_hold = 17000
is_first = True
while len(gpu_available) == 0:
    if not is_first:
        time.sleep(polling_interval['fail'])
    else:
        is_first = False
    gpu_space_available = gpu_state(gpus, get_return=True)
    for gpu_id, space in gpu_space_available.items():
        if space >= space_hold:
            gpu_available = gpu_id
            break
send_mail()
