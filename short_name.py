import shortuuid
import uuid

path='/data/track-ml/eramia/phi025-025_eta025-025_train1_allhits_120320_v2.csv'

u = uuid.uuid4()
print(type(u))

shortuuid.set_alphabet("0123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
shortuuid.ShortUUID().random(length=16)

hostnames = ['/data/track-ml/eramia/phi025-025_eta025-025_train1_allhits_120320_v2.csv',
            '/data/track-ml/eramia/10hits_without_etaphi_filter_30000hits.csv',
            'www.doughellmann.com', 'blog.doughellmann.com']

for name in hostnames:
    print(name)
    uid = uuid.uuid5(uuid.NAMESPACE_DNS, name)
    #uid = uuid.uuid4()
    #uid = uuid.uuid3(uuid.NAMESPACE_DNS, name)
    #uid = shortuuid.uuid(name=name)

    #enc = shortuuid.encode(uid)
    #dec = shortuuid.decode(enc)
    #uid = shortuuid.uuid(name=name)
    enc = shortuuid.encode(uid)
    dec = shortuuid.decode(enc)
    
    print('\tUUID  :', uid)
    print('\tMD5 enc   :', enc)
    print('\tMD5 dec   :', dec)
    if dec == uid: 
    	print(True)
    else: 
    	print(False)