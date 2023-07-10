import cv2
from skimage import exposure,metrics
from skimage import color
import skimage.measure
from skimage.util import img_as_ubyte
import math
import numpy as np
import sewar
from pywt import dwt2
from pywt import idwt2
from contrast_image import contrast_image,quantitation
from tkinter import *
from tkinter import ttk
from tkinter import messagebox 
from tkinter import filedialog
from PIL import Image, ImageTk
import pygad
import gari
import matplotlib.pyplot as plt

root = Tk()
root.title("CELCI")
root.iconphoto(False, ImageTk.PhotoImage(file="Icon.png"))
root.geometry("900x600")
root.minsize(900,600)
root.resizable(width = True, height = True)

image=cv2.imread("p3.png",0)

def resze(i):
    try:
        global image
        image=cv2.resize(i,(300,300))
    except Exception as e:
        print(str(e))
    global img
    img=image

D1=image

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
 
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
 
    return cv2.LUT(src, table)

def mse(org,en):
    difference_array = np.subtract(org,en)
    squared_array = np.square(difference_array)
    return squared_array.mean()

def edge_pixels(input):
    input =input.astype("uint8")
    # Load image, grayscale, Otsu's threshold    
    thresh = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours and perform contour approximation
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:       
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        edge_pixels=len(approx)
        return edge_pixels    
    
def get_num_pixels(input):
    print(input.shape)
    input=D1
    h,w=input.shape
    return h*w

def edge_contentGamma(input):
    edge_content=edge_pixels(input)/get_num_pixels(input)
    return edge_content

def edge_contentCLAHE(input):

    img = cv2.GaussianBlur(input,(3,3),0)

    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sob = np.hypot(sx, sy)

    ans=np.sum(sob)
    ans1=math.log(math.log(ans))
    value=edge_pixels(input)/get_num_pixels(input)
    edge_content=ans1*value
    return edge_content

def std(input):
    std=np.std(input)
    return std

def entropy(input):
    entropy = skimage.measure.shannon_entropy(input)
    return entropy

def obj_funcGamma(input):
    target_chromosome=input
    if input is target_chromosome:
        obj_func=(1/23)*entropy(input)+(1/23)*edge_contentGamma(input)
        # print("Objective Function(Gamma):",obj_func)
        return obj_func

def obj_funcCLAHE(input):
    target_chromosome=input
    if input is target_chromosome:
        obj_func=(1/1200)*entropy(input)+(1/1200)*edge_contentCLAHE(input)+(1/1200)*std(input)
        # print("Objective Function(CLAHE):",obj_func)
        return obj_func

target_chromosome = gari.img2chromosome(image)

def imageinto1D(LL):   
    global target_chromosome 
    target_chromosome = gari.img2chromosome(LL)
    # return target_chromosome

def fitness_fun(solution, solution_idx):
    fitness=obj_funcGamma(target_chromosome-solution)
    fitness=obj_funcGamma(target_chromosome)
    return fitness

def fitness_fun1(solution, solution_idx):
    fitness=obj_funcCLAHE(target_chromosome-solution)
    fitness=obj_funcCLAHE(target_chromosome)
    return fitness

def GA(fitness_fun):    
    ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=10,
                       fitness_func=fitness_fun,
                       sol_per_pop=50,
                       num_genes=target_chromosome.size,
                       init_range_low=0.0,
                       init_range_high=3.0,
                       mutation_percent_genes=0.2,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=3.0)
  
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    return solution_fitness

def qm(psnr,msquare,ssim,rase,sam,AM):
    ag = Toplevel()
    ag.title("Qualitative Measures")
    ag.geometry("300x200")
    ag.minsize(300,200)
    tv = ttk.Treeview(ag)
    tv['columns']=('Name', 'Value')

    tv.column('#0', width=0, stretch=NO)
    tv.column('Name', anchor=CENTER, width=80)
    tv.column('Value', anchor=CENTER, width=150)

    tv.heading('#0', text='', anchor=CENTER)
    tv.heading('Name', text='Name', anchor=CENTER)
    tv.heading('Value', text='Value', anchor=CENTER)  

    tv.insert(parent='', index=0, iid=0, text='', values=('PSNR',psnr))
    tv.insert(parent='', index=1, iid=1, text='', values=('MSE',msquare))
    tv.insert(parent='', index=2, iid=2, text='', values=('SSIM',ssim))
    tv.insert(parent='', index=3, iid=3, text='', values=('RASE',rase))
    tv.insert(parent='', index=4, iid=4, text='', values=('SAM',sam))
    tv.insert(parent='', index=5, iid=5, text='', values=('AM',AM))
    plt.show()
    tv.pack()
    # root.mainloop()

def ce_techniques(o):
     option=o

     if (option=="1"):
            image_hist=cv2.equalizeHist(image)
            # cv2.imshow("Original Image",image)
            # cv2.imshow("Global Histogram Enhancement",image_hist)
            # cv2.imwrite("image_HE.jpg",image_hist)
            psnr=metrics.peak_signal_noise_ratio(image,image_hist)
            msquare=mse(image,image_hist)
            ssim=metrics.structural_similarity(image,image_hist)
            rase=sewar.rase(image,image_hist,ws=1.5)
            sam=sewar.sam(image,image_hist)
            AM=quantitation.Quantitation()
            AM=AM.AMBE(image,image_hist)  
            qm(psnr,msquare,ssim,rase,sam,AM) 
            # cv2.imshow("Original Image",image)
            # cv2.imshow("Global Histogram Enhancement",image_hist)
            # combined = np.hstack((image,image_hist))
            # setup the figure
            fig = plt.figure("Enhanced output")
            # plt.suptitle("Orignal image vs Enhanced Image")
            # show first image
            ax = fig.add_subplot(1, 2, 1)
            ax.title.set_text("Orignal Image")
            plt.imshow(image, cmap="gray")
            plt.axis("off")
            # show the second image
            ax1 = fig.add_subplot(1, 2, 2)
            ax1.title.set_text("Enhanced Image")
            plt.imshow(image_hist, cmap="gray")
            plt.axis("off")
            # show the images
            plt.show()
            # cv2.imshow("original Image(Left) and Enhanced Image(Right)",combined)
            cv2.waitKey()
            cv2.destroyAllWindows()

     elif (option=="2"):
         ahe_image=exposure.equalize_adapthist(image)
        #  ahe_image=np.uint64(ahe_image)
         ahe_image=img_as_ubyte(ahe_image)
         print(image.dtype)
         print(ahe_image.dtype)
         print(image)
         print(ahe_image)
        #  cv2.imshow("Original Image",image)
        #  cv2.imshow("Adaptive histogram enhancement",ahe_image)
        #  combined = np.hstack((image,ahe_image))
        #  cv2.imshow("original Image(Left) and Enhanced Image(Right)",combined)
         psnr=metrics.peak_signal_noise_ratio(image,ahe_image)
         AM=quantitation.Quantitation()
         AM=AM.AMBE(image,ahe_image)   
         ahe_image=np.uint8(ahe_image)
         msquare=mse(image,ahe_image)
         ssim=metrics.structural_similarity(image,ahe_image)
         rase=sewar.rase(image,ahe_image,ws=1.5)
         sam=sewar.sam(image,ahe_image)
         qm(psnr,msquare,ssim,rase,sam,AM) 
         print(f"PSNR value is {psnr} dB")
         print(f"MSE value is {msquare}")
         print(f"SSIM value is {ssim}")
         print(f"Rase va lue is {rase}")
         print(f"SAM value is {sam}")
         print(f"AMBE value is {AM}")
         fig = plt.figure("Enhanced output")
        #  plt.suptitle("Orignal image vs Enhanced Image")
         # show first image
         ax = fig.add_subplot(1, 2, 1)
         ax.title.set_text("Orignal Image")
         plt.imshow(image,cmap="gray")
         plt.axis("off")
         # show the second image
         ax1 = fig.add_subplot(1, 2, 2)
         ax1.title.set_text("Enhanced Image")
         plt.imshow(ahe_image,cmap="gray")
         plt.axis("off")
         # show the images
         plt.show()
         cv2.waitKey()
         cv2.destroyAllWindows()
         

     elif (option=="3"):
            cl3=cv2.createCLAHE(clipLimit=2.9)
            image_cl3=cl3.apply(image)
            psnr3=metrics.peak_signal_noise_ratio(image,image_cl3)
            ssim3=metrics.structural_similarity(image,image_cl3)
            mse3=mse(image,image_cl3)
            rase3=sewar.rase(image,image_cl3,ws=2)
            sam3=sewar.sam(image,image_cl3)
            AM2=quantitation.Quantitation()
            AM2=AM2.AMBE(image,image_cl3) 
            qm(psnr3,mse3,ssim3,rase3,sam3,AM2)
            print(f"PSNR value with clip limit=2.9 is {psnr3} dB")
            print(f"MSE value with clip limit=2.9 is {mse3}")
            print(f"SSIM value with clip limit=2.9 is {ssim3}")
            print(f"Rase value with clip limit=2.9 is {rase3}")
            print(f"SAM value with clip limit=2.9 is {sam3}")
            print(f"AMBE value with clip limit=2.9 is {AM2}")
            fig = plt.figure("Enhanced output")
            # plt.suptitle("Orignal image and Enhanced image")
            # show first image
            ax = fig.add_subplot(1, 2, 1)
            ax.title.set_text("Orignal Image")
            plt.imshow(image, cmap = plt.cm.gray)
            plt.axis("off")
            # show the second image
            ax1 = fig.add_subplot(1, 2, 2)
            ax1.title.set_text("Enhanced Image")
            plt.imshow(image_cl3, cmap = plt.cm.gray)
            plt.axis("off")
            # show the images
            plt.show()
            cv2.waitKey()
            cv2.destroyAllWindows()
            # ce_techniques()
                
     elif (option=="4"):  
            Gamma_co=exposure.adjust_gamma(image,gamma=1.7)
            print(image.dtype)
            print(Gamma_co.dtype)
            print(image)
            print(Gamma_co)
            # cv2.imshow("Original Image",image)
            # cv2.imshow("Gamma correction enhancement",Gamma_co)
            # cv2.imwrite("image_gamma.jpg",Gamma_co)
            psnr=metrics.peak_signal_noise_ratio(image,Gamma_co)
            ssim=metrics.structural_similarity(image,Gamma_co)
            msquare=mse(image,Gamma_co)
            rase=sewar.rase(image,Gamma_co,ws=1.5)
            sam=sewar.sam(image,Gamma_co)
            AM=quantitation.Quantitation()
            AM=AM.AMBE(image,Gamma_co)
            qm(psnr,msquare,ssim,rase,sam,AM) 
            print(f"PSNR value is {psnr} dB")
            print(f"MSE value is {msquare} ")
            print(f"SSIM value is {ssim} ")
            print(f"RASE value is {rase} ")
            print(f"SAM value is {sam} ")
            print(f"AMBE value is {AM} ")
            fig = plt.figure("Enhanced output")
            # plt.suptitle("Orignal image vs Enhanced Image")
            # show the first image
            ax = fig.add_subplot(1, 2, 1)
            ax.title.set_text("Orignal Image")
            plt.imshow(image, cmap = plt.cm.gray)
            plt.axis("off")
            # show the second image
            ax1 = fig.add_subplot(1, 2, 2)
            ax1.title.set_text("Enhanced Image")
            plt.imshow(Gamma_co, cmap = plt.cm.gray)
            plt.axis("off")
            plt.show()
            cv2.waitKey()
            cv2.destroyAllWindows()

     elif (option=="5"):
            img=cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE)
            bb=contrast_image.CI(img)
            bb=bb.BBHE()
            imgGray = color.rgb2gray(bb)
            imgGray=img_as_ubyte(imgGray)
            # print(image.dtype)
            # print(imgGray.dtype)
            # print(image)
            # print(imgGray)
            # cv2.imshow("Original Image",img)
            # cv2.imshow("BPDHE",imgGray)
            # cv2.imwrite("image_BPHE.jpg",imgGray)
            psnr=metrics.peak_signal_noise_ratio(image,imgGray)
            ssim=metrics.structural_similarity(image,imgGray)
            sam=sewar.sam(image,imgGray)
            AM=quantitation.Quantitation()
            AM=AM.AMBE(image,imgGray) 
            msquare=mse(image,imgGray)
            rase=sewar.rase(image,imgGray,ws=1.3)
            qm(psnr,msquare,ssim,rase,sam,AM) 
            print(f"PSNR value is {psnr} dB")
            print(f"MSE value is {msquare} ")
            print(f"SSIM value is {ssim} ")
            print(f"RASE value is {rase} ")
            print(f"SAM value is {sam} ")
            print(f"AMBE value is {AM} ")
            fig = plt.figure("Enhanced output")
            # plt.suptitle("Orignal image vs Enhanced Image")
             # show the first image
            ax = fig.add_subplot(1, 2, 1)
            ax.title.set_text("Orignal Image")
            plt.imshow(img, cmap = plt.cm.gray)
            plt.axis("off")
            # show the second image
            ax1 = fig.add_subplot(1, 2, 2)
            ax1.title.set_text("Enhanced Image")
            plt.imshow(imgGray, cmap = plt.cm.gray)
            plt.axis("off")
            plt.show()
            cv2.waitKey()
            cv2.destroyAllWindows()


     elif (option=="6"):

            img=cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE)
            bb1=contrast_image.CI(img)
            bb1=bb1.RLBHE()
            imgGray1 = color.rgb2gray(bb1)
            imgGray1=img_as_ubyte(imgGray1)
            # cv2.imshow("Original Image",img)
            # cv2.imshow("RLBHE",imgGray1)
            # cv2.imwrite("image_RLBHE.jpg",imgGray1)
            psnr=metrics.peak_signal_noise_ratio(image,imgGray1)
            ssim=metrics.structural_similarity(image,imgGray1)
            rase=sewar.rase(image,imgGray1,ws=1.4)
            sam=sewar.sam(image,imgGray1)
            AM=quantitation.Quantitation()
            AM=AM.AMBE(image,imgGray1) 
            msquare=mse(image,imgGray1)
            qm(psnr,msquare,ssim,rase,sam,AM) 
            print(f"PSNR value is {psnr} dB")
            print(f"MSE value is {msquare} ")
            print(f"SSIM value is {ssim} ")
            print(f"RASE value is {rase} ")
            print(f"SAM value is {sam} ")
            print(f"AMBE value is {AM} ")
            fig = plt.figure("Enhanced output")
            # plt.suptitle("Orignal image vs Enhanced Image")
             # show the first image
            ax = fig.add_subplot(1, 2, 1)
            ax.title.set_text("Orignal Image")
            plt.imshow(img, cmap = plt.cm.gray)
            plt.axis("off")
            # show the second image
            ax1 = fig.add_subplot(1, 2, 2)
            ax1.title.set_text("Enhanced Image")
            plt.imshow(imgGray1, cmap = plt.cm.gray)
            plt.axis("off")
            plt.show()
            cv2.waitKey()
            cv2.destroyAllWindows()
            
     elif (option=="7"):
            img=cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE)
            bb2=contrast_image.CI(img)
            bb2=bb2.RSIHE(recursive=1.3)
            imgGray2 = color.rgb2gray(bb2)
            imgGray2=img_as_ubyte(imgGray2)
            # cv2.imshow("Original Image",img)
            # cv2.imshow("RSIHE",imgGray2)
            # cv2.imwrite("image_RSIHE.jpg",imgGray2)
            psnr=metrics.peak_signal_noise_ratio(image,imgGray2)
            ssim=metrics.structural_similarity(image,imgGray2)
            rase=sewar.rase(image,imgGray2,ws=1.5)
            sam=sewar.sam(image,imgGray2)
            AM=quantitation.Quantitation()
            AM=AM.AMBE(image,imgGray2) 
            msquare=mse(image,imgGray2)
            qm(psnr,msquare,ssim,rase,sam,AM) 
            print(f"PSNR value is {psnr} dB")
            print(f"MSE value is {msquare} ")
            print(f"SSIM value is {ssim} ")
            print(f"RASE value is {rase} ")
            print(f"SAM value is {sam} ")
            print(f"AMBE value is {AM} ")
            fig = plt.figure("Enhanced output")
            # plt.suptitle("Orignal image vs Enhanced Image")
             # show the first image
            ax = fig.add_subplot(1, 2, 1)
            ax.title.set_text("Orignal Image")
            plt.imshow(img, cmap = plt.cm.gray)
            plt.axis("off")
            # show the second image
            ax = fig.add_subplot(1, 2, 2)
            ax.title.set_text("Enhanced Image")
            plt.imshow(imgGray2, cmap = plt.cm.gray)
            plt.axis("off")
            plt.show()
            cv2.waitKey()
            cv2.destroyAllWindows() 
                
     
def dwt_enhancemnet(o):
    select=o
  
    if (select=="1"):
        img_new = np.float32(image)/256
        h, w = img_new.shape
        coeffs2 = dwt2(img_new, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        LL/=256
        LL=exposure.equalize_adapthist(LL, kernel_size=None, clip_limit=obj_funcCLAHE(LL), nbins=256)
        # LL=exposure.equalize_adapthist(LL, kernel_size=None, clip_limit=0.1, nbins=256)
        coeffs2= LL, (LH, HL, HH)
        single=idwt2(coeffs2, 'bior1.3')
        # single=single*1.5
        single=cv2.resize(single,(w,h))
        img_new=img_as_ubyte(img_new)
        single=img_as_ubyte(single)
        print(img_new.dtype)
        print(single.dtype)
        print(img_new)
        print(single)
        # combined = np.hstack((img_new,single))
        # cv2.imshow("original Image(Left) and Enhanced Image(Right)",combined)
        # cv2.imshow("CLAHE implemented on single level DWT",single)
        psnr=metrics.peak_signal_noise_ratio(img_new,single)
        ssim=metrics.structural_similarity(img_new,single)
        rase=sewar.rase(img_new,single,ws=2)
        sam=sewar.sam(img_new,single)
        AM=quantitation.Quantitation()
        AM=AM.AMBE(img_new,single)
        # single=np.uint8(single)
        msquare=mse(img_new,single) 
        qm(psnr,msquare,ssim,rase,sam,AM)
        fig = plt.figure("Enhanced output")
        # plt.suptitle("Orignal image vs Enhanced Image")   
         # show the first image
        ax = fig.add_subplot(1, 2, 1)
        ax.title.set_text("Orignal Image")
        plt.imshow(img_new, cmap = plt.cm.gray)
        plt.axis("off")
        # show the second image
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.title.set_text("Enhanced Image")
        plt.imshow(single, cmap = plt.cm.gray)
        plt.axis("off")
        plt.show()
        print(f"PSNR value is {psnr} dB")
        print(f"MSE value is {msquare}")
        print(f"SSIM value is {ssim}")
        print(f"rase value is {rase}")
        print(f"Sam value is {sam}")
        print(f"AMBE value is {AM}")
        cv2.waitKey()
        cv2.destroyAllWindows()

    elif(select=="2"):
        img_new = np.float32(image)/256
        h, w = img_new.shape
        coeffs2 = dwt2(img_new, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        LL/=256
        LL=exposure.equalize_adapthist(LL, kernel_size=None, clip_limit=0.00011, nbins=255)
        LL1=  dwt2(LL, 'bior1.3')
        LL2,(LH2,HL2,HH2)=LL1
        LL2=LL2/256
        LL2=exposure.equalize_adapthist(LL2, kernel_size=None, clip_limit=0.0009, nbins=255)
        LL2= LL2,(LH2, HL2, HH2)
        LL1=idwt2(LL2, 'bior1.3')
        LL=LL1
        # LL=LL*1.4
        coeffs2= LL, (LH, HL, HH)
        single=idwt2(coeffs2, 'bior1.3')
        # single=single*1.5
        single=cv2.resize(single,(w,h))
        img_new=img_as_ubyte(img_new)
        single=img_as_ubyte(single)
        # print(img_new.dtype)
        # print(Reverse.dtype)
        # print(img)
        # print(img_new)
        # globle1 = np.hstack((img_new,single))
        # combined = np.hstack((img_new,single))
        # cv2.imshow("original Image(Left) and Enhanced Image(Right)",combined)
        # cv2.imshow("CLAHE implemented on single level DWT",single)
        # print(single.shape)
        # print(img_new.shape)
        psnr=metrics.peak_signal_noise_ratio(img_new,single)
        ssim=metrics.structural_similarity(img_new,single)
        rase=sewar.rase(img_new,single,ws=2)
        sam=sewar.sam(img_new,single)
        AM=quantitation.Quantitation()
        AM=AM.AMBE(img_new,single)
        # single=np.uint8(single)
        msquare=mse(img_new,single) 
        qm(psnr,msquare,ssim,rase,sam,AM)   
        print(f"PSNR value is {psnr} dB")
        print(f"MSE value is {msquare}")
        print(f"SSIM value is {ssim}")
        print(f"rase value is {rase}")
        print(f"Sam value is {sam}")
        print(f"AMBE value is {AM}")
        fig = plt.figure("Enhanced output")
        # plt.suptitle("Orignal image vs Enhanced Image")
         # show the first image
        ax = fig.add_subplot(1, 2, 1)
        ax.title.set_text("Orignal Image")
        plt.imshow(img_new, cmap = plt.cm.gray)
        plt.axis("off")
        # show the second image
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.title.set_text("Enhanced Image")
        plt.imshow(single, cmap = plt.cm.gray)
        plt.axis("off")
        plt.show()
        cv2.waitKey()
        cv2.destroyAllWindows()
                
    if (select=="3"):
        img_new = np.float32(image)/256
        h, w = img_new.shape
        coeffs2 = dwt2(img_new, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        # LL/=256
        # LL=img_as_ubyte(LL)
        LL=exposure.adjust_gamma(LL,0.2)
        coeffs2= LL, (LH, HL, HH)
        single=idwt2(coeffs2, 'bior1.3')
        single=cv2.resize(single,(w,h))
        img_new=img_as_ubyte(img_new)
        single=img_as_ubyte(single)
        # print(img_new.dtype)
        # print(single.dtype)
        # print(img)
        # print(img_new)
        # combined = np.hstack((img_new,single))
        # cv2.imshow("original Image(Left) and Enhanced Image(Right)",combined)
        # cv2.imshow("original image",img_new)
        psnr=metrics.peak_signal_noise_ratio(img_new,single)
        ssim=metrics.structural_similarity(img_new,single)
        rase=sewar.rase(img_new,single,ws=2)
        sam=sewar.sam(img_new,single)
        AM=quantitation.Quantitation()
        AM=AM.AMBE(img_new,single)
        # single=np.uint8(single)
        msquare=mse(img_new,single)   
        qm(psnr,msquare,ssim,rase,sam,AM) 
        print(f"PSNR value is {psnr} dB")
        print(f"MSE value is {msquare}")
        print(f"SSIM value is {ssim}")
        print(f"rase value is {rase}")
        print(f"Sam value is {sam}")
        print(f"AMBE value is {AM}")
        fig = plt.figure("Enhanced output")
        # plt.suptitle("Orignal image vs Enhanced Image")
         # show the first image
        ax = fig.add_subplot(1, 2, 1)
        ax.title.set_text("Orignal Image")
        plt.imshow(img_new, cmap = plt.cm.gray)
        plt.axis("off")
        # show the second image
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.title.set_text("Enhanced Image")
        plt.imshow(single, cmap = plt.cm.gray)
        plt.axis("off")
        plt.show()
        cv2.waitKey()
        cv2.destroyAllWindows()

    if(select=="4"):
        img_new = np.float64(image)/256
        # img_new=img_as_float(image)
        h, w = img_new.shape
        print(w)
        coeffs2 = dwt2(img_new, 'bior1.3')
        global D1
        LL, (LH, HL, HH) = coeffs2
        LL=LL/256
        D1=LL
        imageinto1D(D1)
        value=fitness_fun1
        best_value=GA(value)
        LL1=exposure.equalize_adapthist(LL, kernel_size=None, clip_limit=best_value, nbins=256)
        imageinto1D(D1)
        value1=fitness_fun
        best_value1=GA(value1)
        LL=exposure.adjust_gamma(LL1,gamma=best_value1)
        coeffs2=LL,(LH, HL, HH)
        Reverse=idwt2(coeffs2, 'bior1.3')
        Reverse=cv2.resize(Reverse,(w,h))

        img_new=img_as_ubyte(img_new)
        Reverse=img_as_ubyte(Reverse)

        psnr=metrics.peak_signal_noise_ratio(img_new,Reverse)
        ssim=metrics.structural_similarity(img_new,Reverse)
        rase=sewar.rase(img_new,Reverse,ws=2)
        sam=sewar.sam(img_new,Reverse)
        AM=quantitation.Quantitation()
        AM=AM.AMBE(img_new,Reverse)   
        msquare=mse(img_new,Reverse)
        qm(psnr,msquare,ssim,rase,sam,AM) 
        fig = plt.figure("Enhanced output")
        # plt.suptitle("Orignal image vs Enhanced Image")
         # show the first image
        ax = fig.add_subplot(1, 2, 1)
        ax.title.set_text("Orignal Image")
        plt.imshow(img_new, cmap = plt.cm.gray)
        plt.axis("off")
        # show the second image
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.title.set_text("Enhanced Image")
        plt.imshow(Reverse, cmap = plt.cm.gray)
        plt.axis("off")
        plt.show()
        cv2.waitKey()
        cv2.destroyAllWindows()
                                     
def conversion(i):

    # globalimage
    image = np.asarray(i)
    print(image.shape)
    # image=cv2.imread(image,cv2.IMREAD_GRAYSCALE) 
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR) 
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    dimage = Image.fromarray(image)
    dimage = dimage.resize((320, 320))
    ima = ImageTk.PhotoImage(dimage)
    lbl.configure(image=ima)
    # lbl.image=dimage
    resze(image)
    root.mainloop()
# try:
#     image=cv2.resize(image,(555,472))
# except Exception as e:
#     print(str(e))
# global img
# img=image

def open_img():
    filename = filedialog.askopenfilename()
    c = Image.open(filename)
    if c.mode=="RGB":
        m = messagebox.askyesno("Objection",message="The given image is RGB format, do you want this image in gray scale?")
        if(m==1):
            conversion(c)
    else:
        print(c.mode)
        c=c.convert("RGB")
        conversion(c)

# global option
# option=0

def page2():
    pag = Toplevel()
    pag.geometry("700x550")
    pag.title("Traditional Contrast Enhancement")
    pag.iconphoto(True, ImageTk.PhotoImage(file="Icon.png"))
    f1 = Frame(pag, bg="black")
    f1.pack(side=LEFT, fill="y")
    # f2 = Frame(pag)
    # f2.pack(side=TOP,fill="x")
    m = Label(f1, image=photo)
    m.pack(pady=30)
    l = Label(f1, text="Traditional Contrast Enhancement",bg="black",fg="white",font="Helvetica 13")
    l.pack(pady=25)
    Button(pag,text='Histogram equalization',bg="black",fg="white",font="Helvetica 11 bold",padx=75,command=lambda: ce_techniques("1")).pack(pady=20)
    Button(pag,text='Adaptive Histogram equalization',bg="black",fg="white",font="Helvetica 11 bold",padx=43,command=lambda: ce_techniques("2")).pack()
    Button(pag,text='CLAHE',bg="black",fg="white",font="Helvetica 11 bold",padx=133,command=lambda: ce_techniques("3")).pack(pady=25)
    Button(pag,text='Gamma Correction',bg="black",fg="white",font="Helvetica 11 bold",padx=93,command=lambda: ce_techniques("4")).pack()
    Button(pag,text='Brightness Bi-Histogram Equalization',bg="black",fg="white",font="Helvetica 11 bold",padx=28,command=lambda: ce_techniques("5")).pack(pady=25)
    Button(pag,text='Range Limited Bi-Histogram Equalization',bg="black",fg="white",padx=17,font="Helvetica 11 bold",command=lambda: ce_techniques("6")).pack()
    Button(pag,text='Recursive Sub-Image Histogram Equalization',bg="black",fg="white",font="Helvetica 11 bold",command=lambda: ce_techniques("7")).pack(pady=25)
    Button(pag,text="Back",bg="black",fg="white",font="Helvetica 11 bold",padx=143,command=pag.destroy).pack()

def page3():
    pag1 = Toplevel(root)
    pag1.title("Contrast Enhancement using DWT")
    pag1.iconphoto(False, ImageTk.PhotoImage(file="Icon.png"))
    f1 = Frame(pag1, bg="black")
    f1.pack(side=LEFT, fill="y")
    m = Label(f1, image=photo)
    m.pack(pady=30)
    l = Label(f1, text="Contrast Enhancement using DWT",bg="black",fg="white",font="Helvetica 13")
    l.pack(pady=25)
    pag1.geometry("700x550")
    Button(pag1,text='1D level Decompostion and Reconstruction',bg="black",fg="white",font="Helvetica 11 bold",padx=30,command=lambda: dwt_enhancemnet("1")).pack(pady=25)
    Button(pag1,text='2D level Decompostion and Reconstruction',bg="black",fg="white",font="Helvetica 11 bold",padx=30,command=lambda: dwt_enhancemnet("2")).pack()
    Button(pag1,text='1D level Decompostion and Reconstruction Gamma',bg="black",fg="white",font="Helvetica 11 bold",command=lambda: dwt_enhancemnet("3")).pack(pady=25)
    Button(pag1,text='''Gamma Correction applied on CLAHE using GA
    (Proposed Technique)''',bg="black",fg="white",font="Helvetica 11 bold",padx=18,command=lambda: dwt_enhancemnet("4")).pack()
    Button(pag1,text="Back",font="Helvetica 11 bold",bg="black",fg="white",padx=166,command=pag1.destroy).pack(pady=25)

f1 = Frame(root, bg="black")
f1.pack(side=LEFT, fill="y")
# f2 = Frame(root,bg="gray")
# f2.pack(side=TOP,fill="x")
photo=ImageTk.PhotoImage(Image.open("logo.png").resize((100, 100), Image.ANTIALIAS))
m = Label(f1, image=photo)
m.pack(pady=40)
l = Label(f1, text='''  Contrast Enhancement of Low
Contrast Images''',bg="black",fg="white",font="Helvetica 12")
l.pack(pady=8)
la = Label(root, text="Welcome to the Contrast Enhancement Studio", font="Helvetica 16 bold")
la.pack(pady=40)
selectButton= Button(root,text="Browse image",bg="black",fg="white",font="Helvetica 12 bold",command =lambda: open_img()).pack()
lbl=Label(root)
lbl.pack()

Label(root,text="Select any method",font="Helvetica 13 bold").pack(pady=10)
select= Button(root,text="Traditional Contrast Enhancement",bg="black",fg="white",font="Helvetica 12 bold",command=page2).pack()
select1= Button(root,text='''Contrast Enhancement using DWT
(Proposed Method)''',font="Helvetica 12 bold",bg="black",fg="white",command=page3).pack(pady=10)
root.mainloop()