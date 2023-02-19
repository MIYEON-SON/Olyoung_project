from django.shortcuts import render, get_object_or_404
#!pip install json
import json

def index(request):
    return render(request, 'olyoung/index.html')

def ver1(request):

    return render(request, 'olyoung/ver1.html')

def ver1_1(request):
    emp_list=['1','2','3','4','5','6']
    if request.method == 'POST':  #request.POST.get('name', '') -> 변수값이 없다면 default 로 빈값을 임의로 넣어준다.
        skintype=[]
        for i in range(1,7):
            if request.POST.get('skintype_option', None) == 'on':
                skintype.append(request.POST.get('skintype_option', ''))

        # 넘겨줄 데이터 Json 변환
        targetdict = {
            'skintype1' : skintype,
        }

        targetJson = json.dumps(targetdict)

        return render(request, 'olyoung/ver1_1.html',{'skintype1': skintype,'targetJson':targetJson})

    else:
        return render(request, 'olyoung/ver1_1.html', {'skintype1': emp_list})

def ver2(request):
    return render(request, 'olyoung/ver2_result.html')