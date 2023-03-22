import tkinter

#Main window：

window = tkinter.Tk()

window.title("Lite program")

window.resizable(width=False, height=False)# The window size cannot be changed

window.geometry("1000x800+650+100")

#Top-level menu, displayed at the top of the window

menubar= tkinter.Menu(window)

#fmenu can be understood as a menu container for add menu items

fmenu1=tkinter.Menu(window, tearoff=True)#tearoff=True Indicates that the menu can be dragged out

fmenu1.add_separator()#Dividing line

fmenu1.add_command(label='menu1-1')

fmenu1.add_separator()#Dividing line

fmenu1.add_command(label='menu1-2')

fmenu1.add_separator()#Dividing line

fmenu1.add_command(label='menu1-3')

fmenu2=tkinter.Menu(window)

fmenu2.add_separator()#Dividing line

fmenu2.add_command(label='menu2-1')

fmenu3= tkinter.Menu(window)

fmenu3.add_separator()

fmenu3.add_command(label='menu3-1')

fmenu3.add_separator()

fmenu3.add_command(label='menu3-2')

fmenu4=tkinter.Menu(window)#Created a fourth menu container, add four menu containers, to achieve multi-level submenu

fmenu4_1=tkinter.Menu(window)

fmenu4_1.add_command(label='menu4-submenu1-1')

fmenu4_1.add_command(label='menu4-submenu1-2')

fmenu4_2=tkinter.Menu(window)

fmenu4_2.add_command(label='menu4-submenu2-1')

fmenu4_2.add_command(label='menu4-submenu2-2')

fmenu4_3=tkinter.Menu(window)

fmenu4_3.add_command(label='menu4-submenu3-1')

fmenu4_3.add_command(label='menu4-submenu3-2')

fmenu4_4=tkinter.Menu(window)

fmenu4_4.add_command(label='menu4-submenu4-1')

fmenu4_4.add_command(label='menu4-submenu4-2')

#Add fmenu4_1, fmenu4_2, fmenu4_3, fmenu4_4 four menu containers to the fmenu4 menu container

fmenu4.add_cascade(label='menu4-submenu1', menu=fmenu4_1)

fmenu4.add_cascade(label='menu4-submenu2', menu=fmenu4_2)

fmenu4.add_cascade(label='menu4-submenu3', menu=fmenu4_3)

fmenu4.add_cascade(label='menu4-submenu4', menu=fmenu4_4)

#Add four menu containers "fmenu1, fmenu2, fmenu3, fmenu4" to the top-level menu,
# and set the label of the menu container

menubar.add_cascade(label='menu1',menu=fmenu1)

menubar.add_cascade(label='menu2',menu=fmenu2)

menubar.add_cascade(label='menu3',menu=fmenu3)

menubar.add_cascade(label='menu4',menu=fmenu4)

window['menu']= menubar#Set the menu of the window to menubar

window.mainloop()