import matplotlib.pyplot as plt
import os.path
import cv2
import statistics as st
import time
import numpy as np
from numpy import linalg as la
import sys
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QLabel
from PyQt5.QtGui import QIcon
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
# matplotlib.use('QT5Agg')
nrPixeli=10304
nrPersoane=40
nrPozeAntrenare=6
caleBD=r'att_faces'
caleBDpoza_cautata=caleBD+'\s8\8.pgm'
qtcreator_file  = "GUI.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)
datasetDB=1
algoritmFolosit='NN'
normaFolosita='2'
nrTotalPozeAntrenare=nrPersoane*nrPozeAntrenare
A=np.zeros([nrPixeli,nrTotalPozeAntrenare])
norme=['1','2','i','c']

poza_cautata=np.array(cv2.imread(caleBDpoza_cautata,0))
# plt.imshow(poza_cautata, cmap ='gray')
# plt.show()
pozaCautataVector=poza_cautata.reshape(nrPixeli,)
# print(pozaCautataVector)
B=A

def dbConfigurator(button):
    global nrPozeAntrenare
    if(button.text()=='60% training, 40% testing'):
        nrPozeAntrenare=6
    if(button.text()=='80% training, 20% testing'):
        nrPozeAntrenare=8
    if(button.text()=='90% training, 10% testing'):
        nrPozeAntrenare=9

def alegeBD(button):
    global caleBD
    global datasetDB
    if(button.text()=='ORL'):
        caleBD=r'att_faces'
        datasetDB=1
    if(button.text()=='Essex'):
        caleBD=r'att_faces'
        datasetDB=1
    if(button.text()=='CTOVF'):
        caleBD=r'att_faces'
        datasetDB=1

def alegeAlg(button):
    global algoritmFolosit
    if(button.text()=='NN'):
        algoritmFolosit='NN'
    if(button.text()=='KNN, k='):
        algoritmFolosit='KNN'
    if(button.text()=='Eigenfaces, k='):
        algoritmFolosit='Eigen'
    if(button.text()=='Eigenfaces cu RC'):
        algoritmFolosit='EigenRC'
    if(button.text()=='Lanczos, k='):
        algoritmFolosit='Lanczos'

def alegeNorma(button):
    global normaFolosita
    if(button.text()=='Manhattan'):
        normaFolosita='1'
    if(button.text()=='Euclidiana'):
        normaFolosita='2'
    if(button.text()=='Infinit'):
        normaFolosita='i'
    if(button.text()=='Cosinus'):
        normaFolosita='c'

def pozeAntrenare():
    global A, B, nrPersoane, nrTotalPozeAntrenare
    nrTotalPozeAntrenare=nrPersoane*nrPozeAntrenare
    A=np.zeros([nrPixeli,nrTotalPozeAntrenare])
    k=0
    for i in range(1,nrPersoane+1):
        caleFolderPers=caleBD+'\s'+str(i)+'\\'
        for j in range(1,nrPozeAntrenare+1):
            calePozaAntrenare=caleFolderPers+str(j)+'.pgm'
            pozaAntrenare=np.array(cv2.imread(calePozaAntrenare,0))
            pozaVect=pozaAntrenare.reshape(nrPixeli,)
            A[:,k] = pozaVect
            k+=1
    B=A
    return A

class StatisticiApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.title = 'Statistici'
        self.setupUi(self)
        self.selecteazaPoza.clicked.connect(self.openFileNameDialog)
        self.configureDB.buttonClicked.connect(dbConfigurator)
        self.chooseDB.buttonClicked.connect(alegeBD)
        self.algoritm.buttonClicked.connect(alegeAlg)
        self.normag.buttonClicked.connect(alegeNorma)
        self.cauta.clicked.connect(self.cautaPoza)
        self.statistici.clicked.connect(self.rezolvaStatistici)

    def cautaPoza(self):
        global algoritmFolosit
        knnK = int(self.knnK.currentText())
        eigenK = int(self.eigenK.currentText())
        lanczosK = int(self.lanczosK.currentText())
        A= pozeAntrenare()
        print('poza cautata: ',caleBDpoza_cautata)
        print('algoritm folosit: ',algoritmFolosit)
        print('norma folosita:', normaFolosita)
        print('setul de poze: ', datasetDB)
        print('calea pozelor', caleBD)
        print('numr persoane: ',nrPersoane)
        print('numr poze: ',nrPozeAntrenare)
        poza_cautata=np.array(cv2.imread(caleBDpoza_cautata,0))
        pozaCautataVector=poza_cautata.reshape(nrPixeli,)
        if algoritmFolosit=='NN':
            pozitia = NN(A, pozaCautataVector, normaFolosita)
        elif algoritmFolosit=='KNN':
            pozitia = KNN(A, pozaCautataVector, normaFolosita, knnK)
        elif algoritmFolosit=='Eigen':
            pozitia = EIGEN(A, pozaCautataVector, normaFolosita, eigenK)
        elif algoritmFolosit=='EigenRC':
            pozitia = RClasa(A, pozaCautataVector, normaFolosita, eigenK)
        elif algoritmFolosit=='Lanczos':
            pozitia = NNL(A, pozaCautataVector, normaFolosita, lanczosK)
        else:
            print('nici un algo selectat')
        pozitia = (pozitia//nrPozeAntrenare)+1
        pozitiaPozei = pozitia%nrPozeAntrenare
        if pozitiaPozei == 0:
            pozitiaPozei=1
        print('persoana: ',pozitia)
        print('poza: ',pozitiaPozei)
        cvImg=caleBD+'\s'+str(pozitia)+'\\'+str(pozitiaPozei)+'.pgm'
        self.pozaGasita.setPixmap(QtGui.QPixmap(cvImg))

    def rezolvaStatistici(self):
        global algoritmFolosit
        knnK = int(self.knnK.currentText())
        eigenK = int(self.eigenK.currentText())
        lanczosK = int(self.lanczosK.currentText())
        A= pozeAntrenare()
        print('poza cautata: ',caleBDpoza_cautata)
        print('algoritm folosit: ',algoritmFolosit)
        print('norma folosita:', normaFolosita)
        print('setul de poze: ', datasetDB)
        print('calea pozelor', caleBD)
        print('numr persoane: ',nrPersoane)
        poza_cautata=np.array(cv2.imread(caleBDpoza_cautata,0))
        pozaCautataVector=poza_cautata.reshape(nrPixeli,)
        alg=1
        if algoritmFolosit=='NN':
            alg=1
        elif algoritmFolosit=='KNN':
            alg=2
        elif algoritmFolosit=='Eigen':
            alg=3
        elif algoritmFolosit=='EigenRC':
            alg=4
        elif algoritmFolosit=='Lanczos':
            alg=5
        else:
            print('nici un algo selectat')
        print('alg fol: ', alg, 'baza de date', datasetDB )
        STAT(alg,1, datasetDB)

    def openFileNameDialog(self):
        global caleBDpoza_cautata
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Image Files (*.pgm);; Image Files (*.jpg);; Image Files (*.png)", options=options)
        if fileName:
            caleBDpoza_cautata=fileName
            # pixmap = QPixmap(fileName)
            # self.pozaCautata.addPixmap(pixmap)
            # qimg = QImage(fileName)
            # pixmap = QPixmap.fromImage(qimg)
            # self.pozaCautata.addPixmap(pixmap)
            self.pozaCautata.setPixmap(QtGui.QPixmap(fileName))



def caseAlgo(algo, A,stat_pozaVect, norma, k):
    if algo == 1:
        return NN(A, stat_pozaVect, norma)
    elif algo == 2:
        return KNN(A, stat_pozaVect, norma, k)
    elif algo == 3:
        return EIGEN(A, stat_pozaVect, norma, k)
    elif algo == 4:
        return RClasa(A, stat_pozaVect, norma, k)
    elif algo == 5:
        return NNL(A, stat_pozaVect, norma, k)
    else:
        exit()

def caseAlgoNume(algo):
    cases = {
        1: 'NN',
        2: 'KNN',
        3: 'EIGENFACES',
        4: 'EIGENFACES_RC',
        5: 'LANCZOS'
    }
    return cases.get(algo)
def caseStat(stat):
    cases = {
        1: 'RR',
        2: 'TMI'
    }
    return cases.get(stat)
def caseDB(db):
    cases = {
        1: r'att_faces',
        2: r'ctovf_faces',
        3: r'ctovd_faces'
    }
    return cases.get(db)
def caseDBnume(db):
    cases = {
        1: 'ORL',
        2: 'CTOVF',
        3: 'CTOVD'
    }
    return cases.get(db)
def afisarePlot(numefisier, algn, stat):

    fig, ax = plt.subplots()
    nf = np.loadtxt(numefisier, delimiter=' ')
    plt.xticks(nf[:,0])
    plt.plot(nf[:,0],nf[:,1], label='norma 1')
    plt.plot(nf[:,0],nf[:,2], label='norma 2')
    plt.plot(nf[:,0],nf[:,3], label='norma inf')
    plt.plot(nf[:,0],nf[:,4], label='norma cos')
    ax.legend(loc='upper left', shadow=False, fontsize='x-small')
    plt.xlabel('nivel de trunchiere')
    if algn=='NN':
        plt.xlabel('norma')
    plt.ylabel(stat)
    plt.show()

def STAT(alg=2,stat=1, dataset=1):
    stat=caseStat(stat)
    caleBD=caseDB(dataset)
    dataset=caseDBnume(dataset)
    algn=alg
    alg=caseAlgoNume(alg)
    print('numar poze: ', nrPozeAntrenare)
    numefisier=dataset+ '_' + str(nrPozeAntrenare) + '_' + alg + '_' + stat + '.txt'
    numefisier1=dataset+ '_' + str(nrPozeAntrenare) + '_' + alg + '_' + 'RR' + '.txt'
    numefisier2=dataset+ '_' + str(nrPozeAntrenare) + '_' + alg + '_' + 'TMI' + '.txt'
    if(os.path.isfile(numefisier)):
        afisarePlot(numefisier1, algn, 'RR')
        afisarePlot(numefisier2, algn, 'TMI')
    else:
        nrTotalTeste=nrPersoane*(10-nrPozeAntrenare)
        nrRecunoasteriCorecte = 0
        timpiInterogare=[]
        RR = np.zeros([6,6])
        TMI = np.zeros([6,6])
        if alg=='KNN':
            pas=2
            valk=np.arange(3,10,pas)
            RR = np.zeros([4,4])
            TMI = np.zeros([4,4])
        elif alg=='NN':
            pas=1
            valk=np.arange(1,5,pas)
            RR = np.zeros([4,4])
            TMI = np.zeros([4,4])
        else:
            pas=20
            valk=np.arange(20,121,pas)
            RR = np.zeros([6,4])
            TMI = np.zeros([6,4])
        for k in valk:
            for norma in norme:
                if norma=='1':
                    indiceCol=0
                elif norma=='2':
                    indiceCol=1
                elif norma=='i':
                    indiceCol=2
                else:
                    indiceCol=3
                timpiInterogare=[]
                nrRecunoasteriCorecte = 0
                rr=0
                tmi=0
                for i in range(1,41):
                    stat_caleFolderPers=caleBD+'\s'+str(i)+'\\'
                    for j in range(nrPozeAntrenare+1,11):
                        stat_calePozaAntrenare=stat_caleFolderPers+str(j)+'.pgm'
                        stat_pozaAntrenare=np.array(cv2.imread(stat_calePozaAntrenare,0))
                        stat_pozaVect=stat_pozaAntrenare.reshape(nrPixeli,)
                        t0=time.perf_counter()
                        pozitia = caseAlgo(algn, A, stat_pozaVect, norma, k)
                        t1=time.perf_counter()
                        t = t1-t0
                        timpiInterogare.append(t)
                        pozitia = (pozitia//nrPozeAntrenare)+1
                        if(pozitia==i):
                            nrRecunoasteriCorecte=nrRecunoasteriCorecte+1
                rr=nrRecunoasteriCorecte/nrTotalTeste
                print(f'Rata de recunoastere: {rr:.8f}')
                tmi=st.mean(timpiInterogare)
                # print(f'Timp mediu de interogare:{tmi:.8f}')
                RR[k//pas-1,indiceCol]=rr
                TMI[k//pas-1,indiceCol]=tmi
        valk=valk.reshape(-1,1)
        RR=np.hstack((valk,RR))
        numefisierRR=dataset+ '_' + str(nrPozeAntrenare) + '_' + alg + '_' + stat + '.txt'
        RR=np.savetxt(numefisierRR,RR,fmt="%10.9f")
        TMI=np.hstack((valk,TMI))
        numefisierTMI=dataset+ '_' + str(nrPozeAntrenare) + '_' + alg + '_' + 'TMI.txt'
        TMI=np.savetxt(numefisierTMI,TMI,fmt="%10.9f")
        print('nume fisier ', numefisier1, ', algoritm folosit ',algn)
        print('nume fisier ', numefisier2, ', algoritm folosit ',algn)
        afisarePlot(numefisier1, algn, 'RR')
        afisarePlot(numefisier2, algn, 'TMI')

def NN(A,poza_cautata, norma):
    norma=str(norma)
    poza_cautata=poza_cautata.reshape(nrPixeli,)
    z=np.zeros(([1,len(A[0])]), dtype=float)
    for i in range(0,len(A[0])):
        if norma =='1':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,1)
        elif norma=='2':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,2)
        elif norma=='i':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,np.inf)
        elif norma=='c':
            numarator =np.dot(poza_cautata, A[:,i])
            numitor=la.norm(poza_cautata)*la.norm(A[:,i])
            z[0,i]=(1-numarator)/numitor
        else:
            exit()
    # imagine = A[:, np.argmin(z)]
    # print(imagine)
    # poza=np.reshape(imagine,(112,92))
    # plt.imshow(poza, cmap ='gray')
    # plt.show()
    return np.argmin(z)

def KNN(A,poza_cautata, norma, K):
    poza_cautata=poza_cautata.reshape(nrPixeli,)
    z=np.zeros(([len(A[0])]), dtype=float)
    for i in range(0,len(A[0])):
        if norma =='1':
            diferenta = poza_cautata-A[:,i]
            z[i]=la.norm(diferenta,1)
        elif norma=='2':
            diferenta = poza_cautata-A[:,i]
            z[i]=la.norm(diferenta,2)
        elif norma=='i':
            diferenta = poza_cautata-A[:,i]
            z[i]=la.norm(diferenta,np.inf)
        elif norma=='c':
            numarator =np.dot(poza_cautata, A[:,i])
            numitor=la.norm(poza_cautata)*la.norm(A[:,i])
            z[i]=(1-numarator)/numitor
        else:
            exit()
    pozitii=np.argsort(z)
    pozitii=pozitii[:K]
    vecini=np.zeros(len(pozitii),)
    for i in range(0,len(pozitii)):
        if pozitii[i]%nrPozeAntrenare == 0:
            vecini[i]=pozitii[i]/nrPozeAntrenare
        else:
            vecini[i]=pozitii[i]//nrPozeAntrenare+1
    # print(vecini)
    vecin=int(st.mode(vecini))
    # print(vecin)
    pozitie=nrPozeAntrenare*(vecin-1)
    # poza_gasita=A[:,pozitie]
    # print(poza_gasita)
    # poza_gasita=poza_gasita.reshape(112,92)
    # plt.imshow(poza_gasita,  cmap ='gray')
    # plt.show()
    return pozitie
def EIGEN(A, pozaCautataVector, norma, k):
    t0=time.perf_counter()
    media = np.mean(A, axis=1)
    A=(A.T-media).T
    # C=np.dot(A,A.T)
    L=np.dot(A.T,A)
    d, v = np.linalg.eig(L)
    v=np.dot(A,v)
    vsorted = np.argsort(d)
    vsorted = vsorted[-1:-k-1:-1]
    HQPB = np.zeros([nrPixeli,k])
    for n in range(0,k):
        # print(vsorted[n])
        HQPB[:,n]= v[:,vsorted[n]]
    proiectii=np.dot(A.T, HQPB)
    t1=time.perf_counter()
    t=t1-t0
    # print(t)
    pozacautata = pozaCautataVector-media
    proiectie_cautata=np.dot(pozacautata,HQPB)
    return NNEIG(proiectii.T, proiectie_cautata, norma)


def NNEIG(A,poza_cautata, norma):
   # poza_cautata=poza_cautata.reshape(nrPixeli,)
    z=np.zeros(([1,len(A[0])]), dtype=float)
    for i in range(0,len(A[0])):
        if norma =='1':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,1)
        elif norma=='2':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,2)
        elif norma=='i':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,np.inf)
        elif norma=='c':
            numarator =np.dot(poza_cautata, A[:,i])
            numitor=la.norm(poza_cautata)*la.norm(A[:,i])
            z[0,i]=(1-numarator)/numitor
        else:
            exit()
    # imagine = B[:, np.argmin(z)]
    # print(imagine)
    # poza=np.reshape(imagine,(112,92))
    # plt.imshow(poza, cmap ='gray')
    # plt.show()
    return np.argmin(z)

def RClasa(A, pozaCautataVector, norma, k):
    RC=np.zeros([nrPixeli,nrPersoane])

    for t in range(0,nrPersoane):
        start=t*nrPozeAntrenare
        RC[:,t] = np.mean(A[:,start:start+nrPozeAntrenare], axis=1)

    A=RC
    t0=time.perf_counter()
    media = np.mean(A, axis=1)
    A=(A.T-media).T
    L=np.dot(A.T,A)
    d, v = np.linalg.eig(L)
    v=np.dot(A,v)

    vsorted = np.argsort(d)
    vsorted = vsorted[-1:-k-1:-1]
    HQPB = np.zeros([nrPixeli,k])
    for n in range(0,k):
        HQPB[:,n]= v[:,vsorted[n]]
    proiectii=np.dot(A.T, HQPB)
    t1=time.perf_counter()
    t=t1-t0
    # print(t)

    pozacautata = pozaCautataVector-media
    proiectie_cautata=np.dot(pozacautata,HQPB)

    nn = NNRC(proiectii.T, proiectie_cautata, norma)
    # imagine = B[:, nn * nrPozeAntrenare]
    # poza = np.reshape(imagine, (112, 92))
    # plt.imshow(poza, cmap='gray')
    # plt.show()
    # print(nn)
    return (nn * nrPozeAntrenare)


def NNRC(A,poza_cautata, norma):
   # poza_cautata=poza_cautata.reshape(nrPixeli,)
    z=np.zeros(([1,len(A[0])]), dtype=float)
    for i in range(0,len(A[0])):
        if norma =='1':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,1)
        elif norma=='2':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,2)
        elif norma=='i':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,np.inf)
        elif norma=='c':
            numarator =np.dot(poza_cautata, A[:,i])
            numitor=la.norm(poza_cautata)*la.norm(A[:,i])
            z[0,i]=(1-numarator)/numitor
        else:
            exit()
    return np.argmin(z)

def Lanczos( A, k=20):
    q=np.zeros([nrPixeli,k+2])
    q[:,0]=np.zeros(nrPixeli)
    q[:,1]=np.ones(nrPixeli)
    q[:,1]=q[:,1]/la.norm(q[:,1])
    beta=0
    for i in range(1,k+1):
        w=np.dot(A,np.dot(A.T,q[:,i]))-np.dot(beta, q[:,i-1])
        alpha=np.dot(w,q[:,i])
        w=w-np.dot(alpha,q[:,i])
        beta=la.norm(w,2)
        q[:,i]=w/beta
    return q[:,2:k+1]


def NNL(A,pozaCautataVector, norma, k):
    t0 = time.perf_counter()
    HQPB = Lanczos(A, k)
    proiectii = np.dot(A.T, HQPB)
    A=proiectii.T
    poza_cautata = np.dot(pozaCautataVector, HQPB)
    t1 = time.perf_counter()
    t = t1 - t0
    # print(t)
    # poza_cautata=poza_cautata.reshape(nrPixeli,)
    z=np.zeros(([1,len(A[0])]), dtype=float)
    for i in range(0,len(A[0])):
        if norma =='1':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,1)
        elif norma=='2':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,2)
        elif norma=='i':
            diferenta = poza_cautata-A[:,i]
            z[0,i]=la.norm(diferenta,np.inf)
        elif norma=='c':
            numarator =np.dot(poza_cautata, A[:,i])
            numitor=la.norm(poza_cautata)*la.norm(A[:,i])
            z[0,i]=(1-numarator)/numitor
        else:
            exit()
    # imagine = B[:, np.argmin(z)]
    # print(imagine)
    # poza=np.reshape(imagine,(112,92))
    # plt.imshow(poza, cmap ='gray')
    # plt.show()
    return np.argmin(z)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = StatisticiApp()
    window.show()
    sys.exit(app.exec_())
