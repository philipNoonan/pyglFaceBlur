import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from glob import glob
import cv2
import re
import os
import imgui
from imgui.integrations.glfw import GlfwRenderer
from pathlib import Path

import mediapipe as mp

import openvr



def createTexture(texture, target, internalFormat, levels, width, height, depth, minFilter, magFilter):

    if texture == -1:
        texName = glGenTextures(1)
    else:
        glDeleteTextures(int(texture))
        texName = texture
        texName = glGenTextures(1)

    glBindTexture(target, texName)
    #texture wrapping params
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER )
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER )
    #texture filtering params
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter)
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter)
    if target == GL_TEXTURE_1D:
        glTexStorage1D(target, levels, internalFormat, width)
    elif target == GL_TEXTURE_2D:
        glTexStorage2D(target, levels, internalFormat, width, height)
    elif target == GL_TEXTURE_3D or depth > 1:
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER )
        glTexStorage3D(target, levels, internalFormat, width, height, depth)

    return texName

def reset():
    try:
        cap
    except NameError:
        print('')
    else:
        cap.release()

def openVideo(filename):
    cap = cv2.VideoCapture(filename)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))

    return cap, width, height

def openCamera(camera):
    cap = cv2.VideoCapture(int(camera))
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)


    return cap, width, height

def generateTextures(textureDict, numImages, width, height):

    
    maxLevels = 6 # FIXME
    numLevels = 6 # FIXME TOO

    textureDict['raw'] = createTexture(textureDict['raw'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['blurred'] = createTexture(textureDict['blurred'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

	# Allocate the immutable GPU memory storage -more efficient than mutable memory if you are not going to change image size after creation
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, int(width))

    return textureDict


def main():


    # initialize glfw
    if not glfw.init():
        return
    #glfw.window_hint(glfw.VISIBLE, False)    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    #creating the window
    window = glfw.create_window(1600, 900, "face blurring", None, None)
    if not window:
        glfw.terminate()
        return

    xPos = 0
    yPos = 0

    maxLevels = 6





    glfw.make_context_current(window)


    showFileDialogueOptions = False
    showCameraDialogueOptions = False

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    textureDict = {
        'raw' : -1, 
        'blurred' : -1, 

    }

    bufferDict = {
        'sumFlow' : -1
    }

    #bufferDict = generateBuffers(bufferDict)




    imgui.create_context()
    impl = GlfwRenderer(window)
    #           positions         texture coords
    quad = [   -1.0, -1.0, 0.0,   0.0, 0.0,
                1.0, -1.0, 0.0,   1.0, 0.0,
                1.0,  1.0, 0.0,   1.0, 1.0,
               -1.0,  1.0, 0.0,   0.0, 1.0]

    quad = np.array(quad, dtype = np.float32)

    indices = [0, 1, 2,
               2, 3, 0]

    indices = np.array(indices, dtype= np.uint32)

    vertex_shader = (Path(__file__).parent / 'shaders/screenQuad.vert').read_text()

    fragment_shader = (Path(__file__).parent / 'shaders/screenQuad.frag').read_text()

    screen_quad_shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    # set up VAO and VBO for full screen quad drawing calls
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 80, quad, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 24, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)


    # make some default background color
    glClearColor(0.2, 0.3, 0.2, 1.0)

    frameCounter = 0

    #default to not running any filters
    currentFile = 0
    fileList = []
    listy = list(Path('./data/').glob('./*'))
    for x in listy:
        fileList.append(str(x))

    currentCamera = 0
    cameraList = ['0', '1', '2', '3', '4']
    resetVideoSource = True
    sourceAvailable = False

    global cap

    filemode = 0 # 1 : webcam, 2 : video file

    numberOfImages = 5

    width = 0
    height = 0

    numberOfFrames = 1
    firstFrame = True

    my_eyes=mp_face_mesh.FACE_CONNECTIONS
    writer = None
    last_results = None
    

    #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.05,
        min_tracking_confidence=0.05) as face_mesh:

        while not glfw.window_should_close(window):

            glfw.poll_events()
            impl.process_inputs()
            imgui.new_frame()

            if resetVideoSource:
                if sourceAvailable:
                    reset()
                    if filemode == 1:
                        cap, width, height = openCamera(cameraList[currentCamera])
                        #print("width, height", width, height)

                    elif filemode == 2:
                        cap, width, height = openVideo(fileList[currentFile])
                        #print("width, height", width, height)
                        numberOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)


                    textureDict = generateTextures(textureDict, numberOfImages, width, height)
                    #densifiactionFBO = DIS.generateDensificationFramebuffer(textureList[5], width, height)

                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    writer = cv2.VideoWriter('.//output.wmv', fourcc, 20.0, (int(width),  int(height)), True)

                    resetVideoSource = False

            else:
                ret, frame = cap.read()

                if imgui.is_mouse_clicked():
                    if not imgui.is_any_item_active():
                        mouseX, mouseY = imgui.get_mouse_pos()
                        w, h = glfw.get_framebuffer_size(window)
                        xPos = ((mouseX % int(w / 3)) / (w / 3) * width)
                        yPos = (mouseY / (h)) * height

                        #print(xPos, " ", yPos)

                if ret:

                    glBindFramebuffer(GL_FRAMEBUFFER, 0)

                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, textureDict['raw'])
                    img_data = np.array(frame.data, np.uint8)
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_BGR, GL_UNSIGNED_BYTE, img_data)

                    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False
                    results = face_mesh.process(frame)

                    frame.flags.writeable = True
                    if not results.multi_face_landmarks:
                        results = last_results
                    else:
                        last_results = results
                    for face_landmarks in results.multi_face_landmarks:
                                mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACE_CONNECTIONS,
                                landmark_drawing_spec=drawing_spec,
                                connection_drawing_spec=drawing_spec)
                        
                    
                    glBindTexture(GL_TEXTURE_2D, textureDict['blurred'])

                    img_data = np.array(frame.data, np.uint8)
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_BGR, GL_UNSIGNED_BYTE, img_data)

                    writer.write(img_data)


                    # inputImage = np.array(cv2.resize(((np.array(frame.data, np.float32) / 255.0) - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225), (320, 512)), dtype=np.float32)

                    # res = inference(engine, segContext, inputImage.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
                    # otemp = np.zeros((10, 16, 3))

                    # s_w = 16
                    # s_h = 10
                    # s_c = 21

                    # for y in range(s_h):
                    #     for x in range(s_w):

                    #         p_max = -100000.0
                    #         c_max = -1
                            
                    #         for c in range(1, s_c, 1):

                    #             p = res[c * s_w * s_h + y * s_w + x]
                    #             if( c_max < 0 or p > p_max ):
                    #                 p_max = p
                    #                 c_max = c

                    #         otemp[y,x,:] = classColors[c_max]

                    # #outMask = np.zeros((10, 16)).astype(np.float32)
                    # #outy = np.argmax(otemp, axis=0)

                    # cv2.imshow('frame', otemp)
                    # cv2.resizeWindow('frame', 512,320)
                    # cv2.waitKey(1)

                
                    w, h = glfw.get_framebuffer_size(window)



                    # set the active drawing viewport within the current GLFW window (i.e. we are spliiting it up in 2 cols)
                    xpos = 0
                    ypos = 0
                    xwidth = float(w) / 2.0
                    glViewport(int(xpos), int(ypos), int(xwidth),h)
                    glClear(GL_COLOR_BUFFER_BIT)

                    glUseProgram(screen_quad_shader)

                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, textureDict['raw'])
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                    xpos = xwidth
                    glViewport(int(xpos), int(ypos), int(xwidth),h)
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, textureDict['blurred'])
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)








                elif ret == False:
                    break


            # GUI TIME
            imgui.begin("Menu", True)

            if imgui.button("Use Camera"):
                filemode = 1
                showCameraDialogueOptions = True

            if imgui.button("Select Video File"):
                filemode = 2
                showFileDialogueOptions = True

            if showCameraDialogueOptions:
                clicked, currentCamera = imgui.combo(
                    "cams", currentCamera, cameraList
                )
                if (clicked):
                    resetVideoSource = True
                    sourceAvailable  = True
                    frameCounter = 0

            if showFileDialogueOptions:
                clicked, currentFile = imgui.combo(
                    "files", currentFile, fileList
                )
                if (clicked):
                    resetVideoSource = True
                    sourceAvailable  = True
                    frameCounter = 0


            #changedR, sliderRValue = imgui.slider_int("sliceR", sliderRValue, min_value=0, max_value=5)
            #changedG, frameCounter = imgui.slider_int("frame", frameCounter, min_value=0, max_value=numberOfFrames)

            # #changedB, sliderBValue = imgui.slider_int("sliceB", sliderBValue, min_value=0, max_value=numberOfImages)
            # _, doFilterEnabled = imgui.checkbox("run filter", doFilterEnabled)
            # _, getPose = imgui.checkbox("run pose", getPose)

            # if changedG:
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, frameCounter)

            imgui.end()





            imgui.render()

            impl.render(imgui.get_draw_data())

            #print((time.perf_counter() - sTime) * 1000)


            glfw.swap_buffers(window)

            frameCounter = frameCounter + 2

            if frameCounter >= numberOfFrames:
                frameCounter = 0


    cap.release()

if __name__ == "__main__":
    main()
