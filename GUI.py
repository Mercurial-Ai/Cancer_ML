from tkinter import *
import tkinter.font as tkFont
from tkinter import ttk
import pandas as pd

themeColor = "orange"

class index:
    def setup(self):
        self.window = Tk()
        self.window.title("Cancer ML")
        self.window.iconbitmap("D:\Cancer_Project\Cancer_ML\cancer_icon.ico")

        # font used for page elements
        self.font = tkFont.Font(family="Georgia",size=12)

    # choices of data
    def HNSCC(self):
        self.dataset = "HNSCC"
        self.HNSbutton.config(relief=SUNKEN,background=themeColor)
        self.HN1button.config(relief=RAISED,background="SystemButtonFace")
        self.BRbutton.config(relief=RAISED,background="SystemButtonFace")
        self.otherButton.config(relief=RAISED, background="SystemButtonFace")

    def HN1(self):
        self.dataset = "HN1"
        self.HNSbutton.config(relief=RAISED, background="SystemButtonFace")
        self.HN1button.config(relief=SUNKEN, background=themeColor)
        self.BRbutton.config(relief=RAISED, background="SystemButtonFace")
        self.otherButton.config(relief=RAISED, background="SystemButtonFace")

    def BRIC(self):
        self.dataset = "METABRIC"
        self.HNSbutton.config(relief=RAISED, background="SystemButtonFace")
        self.HN1button.config(relief=RAISED, background="SystemButtonFace")
        self.BRbutton.config(relief=SUNKEN, background=themeColor)
        self.otherButton.config(relief=RAISED, background="SystemButtonFace")

    def other(self):
        self.dataset = "other"
        self.HNSbutton.config(relief=RAISED, background="SystemButtonFace")
        self.HN1button.config(relief=RAISED, background="SystemButtonFace")
        self.BRbutton.config(relief=RAISED, background="SystemButtonFace")
        self.otherButton.config(relief=SUNKEN, background=themeColor)

    def cont(self):
        self.window.destroy()

    def label(self):
        labelFont = tkFont.Font(family="Georgia",size=20)
        label = Label(text="Select a configuration",font=labelFont)
        label.grid(column=1,padx=40,pady=40)

    def draw(self):
        self.label()

        # make buttons
        self.HNSbutton = Button(text="HNSCC Dataset", width=25, height=3, font=self.font, command=self.HNSCC,
                             activebackground=themeColor)
        self.HNSbutton.grid(column=1, pady=25)

        self.HN1button = Button(text="HNSCC-HN1 Dataset", width=25, height=3, font=self.font, command=self.HN1,
                                activebackground=themeColor)
        self.HN1button.grid(column=1,pady=25)

        self.BRbutton = Button(text="METABRIC Dataset", width=25, height=3, font=self.font, command=self.BRIC,
                               activebackground=themeColor)
        self.BRbutton.grid(column=1,pady=25)

        self.otherButton = Button(text="Other", width=25,height=3,font=self.font,command=self.other)
        self.otherButton.grid(column=1,pady=25)

        self.contButton = Button(text="Continue", width=25, height=3,font=self.font,command=self.cont)
        self.contButton.grid(column=1,pady=35)

        self.window.mainloop()

class targetUI:
    def __init__(self,dataset):
        self.data = dataset

    def setup(self):
        self.window = Tk()
        self.window.title("Cancer ML")
        self.window.iconbitmap("D:\Cancer_Project\Cancer_ML\cancer_icon.ico")

    def getCols(self):
        df = pd.read_csv(self.data)
        cols = list(df.columns)
        return cols

    def draw(self):
        cols = self.getCols()

        buttonList = []
        for var in cols:
            var = str(var)

            button = Button(text=var)
            button.grid(column=1,pady=40)

            buttonList.append(button)

indexPage = index()
indexPage.setup()
indexPage.draw()
dataset = indexPage.dataset

if dataset == "HNSCC":
    save_fit = False
    load_fit = False

    main_data = "HNSCC/Patient and Treatment Characteristics.csv"
    sec_data = ""
    test_file = ""

    target_variables = target

with open("D:\Cancer_Project\Cancer_ML\project_variables.txt","r") as projectVars:
    vars=projectVars.readlines()

window = Tk()

window.title("Cancer ML")

window.iconbitmap("D:\Cancer_Project\Cancer_ML\cancer_icon.ico")

main_frame = Frame(window)
main_frame.pack(fill=BOTH,expand=1)

canvas = Canvas(main_frame)
canvas.pack(side=LEFT, fill=BOTH, expand=1)

# Add scrollbars to the canvas
scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

scrollbar_x = ttk.Scrollbar(main_frame,orient=HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=BOTTOM, fill=X)

# Configure the canvas
canvas.configure(xscrollcommand=scrollbar_x.set)
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

second_frame = Frame(canvas)
canvas.create_window((0,0), window=second_frame, anchor="nw")

# get screen dimensions
screenHeight = second_frame.winfo_screenheight()
screenWidth = second_frame.winfo_screenwidth()

# make font for title
fontTitle = tkFont.Font(family="Georgia",size=20)

# make font for variable labels
varFont = tkFont.Font(family="Times New Roman",size=16)

# color for variable labels
varColor = "#262121"

# make font for entry boxes
inputFont = tkFont.Font(family="Georgia",size=12)

# make title
title = Label(second_frame,text="Data Page",font=fontTitle)
title.grid(column=1)

# initialize vars
boolList = []

# class to make pair of true/false buttons
class boolButtons:
    def __init__(self,name):
        self.name = name

    def true(self):
        global boolList

        bool = True
        boolList.append(bool)

        self.buttonTrue.config(relief=SUNKEN,background=themeColor)
        self.buttonFalse.config(relief=RAISED,background='SystemButtonFace')

    def false(self):
        global boolList

        bool = False
        boolList.append(bool)

        self.buttonTrue.config(relief=RAISED,background='SystemButtonFace')
        self.buttonFalse.config(relief=SUNKEN,background=themeColor)

    def label(self):
        label = Label(second_frame,text=self.name,font=varFont,fg=varColor)
        label.grid(column=1,pady=40)

    def makeButton(self):
        self.buttonTrue = Button(second_frame,text="True",width=25,height=3,font=inputFont,command=self.true,activebackground=themeColor)
        self.buttonTrue.grid(column=1)

        self.buttonFalse = Button(second_frame,text="False",width=25,height=3,font=inputFont,command=self.false,activebackground=themeColor)
        self.buttonFalse.grid(column=1)

def makeEntry(name):
    entryText = StringVar()

    label = Label(second_frame,text=name,font=varFont,fg=varColor)
    label.grid(column=1,pady=40)

    entry = Entry(second_frame,textvariable=entryText,font=inputFont,width=25)
    entry.grid(column=1)
    return entryText

# initialize list for entryText vars
entryText_list = []

# initialize lists for var names
varList_txt = []
varList_bool = []

for var in vars:
    var = str(var)
    # checking if variable is a string
    i = var.find('"')

    # check if string contains tuple
    t = var.find("(")
    t2 = var.find(")")

    # check if string contains numbers
    n = any(char.isdigit() for char in var)

    # Only use part of variable before equal sign
    var = var.rsplit('=', 1)[0]

    if i == -1 and t == -1 and t2 == -1 and n == False:
        label = boolButtons(var[:-1]).label()
        buttons = boolButtons(var[:-1]).makeButton()
        varList_bool.append(var)
    else:
        entryText = makeEntry(var[:-1])
        entryText_list.append(entryText)
        varList_txt.append(var)

# make new list for converted tkinter objects
txtEntry_list = []

def Continue():
    global txtEntry_list

    for entries in entryText_list:
        txtEntry = entries.get()
        entries.set("")
        txtEntry_list.append(txtEntry)

    window.quit()

# initialize bool
useDefaults = False

def useDefault():
    global useDefaults

    useDefaults = True
    window.quit()

# make default button
defaultButton = Button(second_frame,text="Use Defaults",bg=themeColor,command=useDefault,font=varFont)
defaultButton.grid(column=2,padx=screenWidth/1.4)

# make continue button
contButton = Button(second_frame,text="Continue",bg=themeColor,command=Continue,font=varFont)
contButton.grid(column=2,padx=screenWidth/1.4)

window.mainloop()