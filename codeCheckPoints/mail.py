import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def sendMail():
    fromaddr = "notes.notifier@gmail.com"
    toaddr = "guptapn@rknec.edu,dholwanimona@gmail.com,varmans_1@rknec.edu,guptava@rknec.edu,deshmukhsp_1@rknec.edu"
    msg = MIMEMultipart() 
    msg['From'] = fromaddr 
    msg['To'] = toaddr 
    msg['Subject'] = "Notes for today's class" 
    body = "PFA notes for today's lecture." 
    msg.attach(MIMEText(body, 'plain')) 
    filename = "output.zip"
    attachment = open("static\\output\\output.zip", "rb") 
    p = MIMEBase('application', 'octet-stream') 
    p.set_payload((attachment).read()) 
    encoders.encode_base64(p) 
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
    msg.attach(p) 
    s = smtplib.SMTP('smtp.gmail.com', 587) 
    s.starttls() 
    s.login(fromaddr, "MakeItHappen") 
    text = msg.as_string() 
    s.sendmail(fromaddr, toaddr.split(","), text) 
    s.quit()






