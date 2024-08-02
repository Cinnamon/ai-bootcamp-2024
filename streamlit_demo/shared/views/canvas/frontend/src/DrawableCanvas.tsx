import React, { useEffect, useState } from "react"
import {
  ComponentProps,
  Streamlit,
  withStreamlitConnection,
} from "streamlit-component-lib"
import { fabric } from "fabric"
import { isEqual } from "lodash"

import CanvasToolbar from "./components/CanvasToolbar"

import { useCanvasState } from "./DrawableCanvasState"
import { tools, FabricTool } from "./lib"

function getStreamlitBaseUrl(): string | null {
  const params = new URLSearchParams(window.location.search)
  const baseUrl = params.get("streamlitUrl")
  if (baseUrl == null) {
    return null
  }

  try {
    return new URL(baseUrl).origin
  } catch {
    return null
  }
}

interface CustomFabricCanvas extends fabric.Canvas {
  isDragging?: boolean;
  selection?: boolean;
  lastPosX?: number;
  lastPosY?: number;

  secondTimeAccess?: boolean;
  currentState?: Object;
  showingMode?: string;

}

/**
 * Arguments Streamlit receives from the Python side
 */
export interface PythonArgs {
  fillColor: string
  strokeWidth: number
  strokeColor: string
  backgroundColor: string
  backgroundImageURL: string
  realtimeUpdateStreamlit: boolean
  canvasWidth: number
  canvasHeight: number
  drawingMode: string
  initialDrawing: Object
  displayToolbar: boolean
  displayRadius: number
  showingMode: string
}

/**
 * Define logic for the canvas area
 */
const DrawableCanvas = ({ args }: ComponentProps) => {
  const {
    canvasWidth,
    canvasHeight,
    backgroundColor,
    backgroundImageURL,
    realtimeUpdateStreamlit,
    drawingMode,
    fillColor,
    strokeWidth,
    strokeColor,
    displayRadius,
    initialDrawing,
    displayToolbar,
    showingMode
  }: PythonArgs = args

  /**
   * State initialization
   */
  const [canvas, setCanvas] = useState<CustomFabricCanvas>(new fabric.Canvas("c") as CustomFabricCanvas);
  canvas.stopContextMenu = true
  canvas.fireRightClick = true

  const [selectedRect, setSelectedRect] = useState(-1)

  const [backgroundCanvas, setBackgroundCanvas] = useState<CustomFabricCanvas>(new fabric.Canvas("c") as CustomFabricCanvas);
  const {
    canvasState: {
      action: { shouldReloadCanvas, forceSendToStreamlit },
      currentState,
      initialState,
    },
    saveState,
    undo,
    redo,
    canUndo,
    canRedo,
    forceStreamlitUpdate,
    resetState,
  } = useCanvasState()

  
  /*
   * Load background image from URL 
   */
  // const params = new URLSearchParams(window.location.search);
  // const baseUrl = params.get('streamlitUrl')
  const baseUrl = getStreamlitBaseUrl() ?? ""
  let img = new fabric.Image()

  fabric.Image.fromURL(baseUrl + backgroundImageURL, function(oImg) {
    img = oImg
    img.selectable = false;
    backgroundCanvas.add(img);

    if (img.width == null || img.height == null){
      return
    }

    // only initialize (image + rects) for canvas 1
    const isSecondTimes = (canvas.secondTimeAccess || false)

    /*
     * This is the first time UI is created, 
     * And we try to align the canvas size with image by perform zooming only.
     * PS: This happend only for 1st time 
     */
    if (isSecondTimes === false){ // It means this is the first time
      console.log("Render Fist Time")
      canvas.loadFromJSON(initialDrawing, () => {})
      
      // initialize zoom
      const widthRatio = canvas.getWidth() / img.width;
      const heightRatio = canvas.getHeight() / img.height;
      const zoom = Math.min(widthRatio, heightRatio)
      canvas.setZoom(zoom);
      backgroundCanvas.setZoom(zoom)

      canvas.secondTimeAccess = true
      canvas.requestRenderAll()
      backgroundCanvas.requestRenderAll()

      canvas.currentState = { ...initialDrawing }
      canvas.showingMode = showingMode
    }

    /*
     * User can choose some group of boxes to visualie (keys only, value only, or both)
     * Refresh the initial canvas
     * The current showingMode is different with the previous one! => Trigger to re-load the initialDrawings!
     * [07.10.2023] The below code should be erased. We don't allow to do it anymore because of low performance.
     */
    if (canvas.showingMode !== showingMode){
      canvas.showingMode = showingMode

      if (!isEqual(canvas.currentState, initialDrawing)){
        canvas.loadFromJSON(initialDrawing, () => {
          canvas.currentState = { ...initialDrawing }
          
          canvas.renderAll()
        })
      }
    }

  });

  /**
   * Initialize canvases on component mount
   * NB: Remount component by changing its key instead of defining deps
   */
  useEffect(() => {
    const c = new fabric.Canvas("canvas", {
      enableRetinaScaling: false,
    })
    const imgC = new fabric.Canvas("backgroundimage-canvas", {
      enableRetinaScaling: false,
    })
    setCanvas(c)
    setBackgroundCanvas(imgC)
    Streamlit.setFrameHeight()
  }, [])


  /**
   * If state changed from undo/redo/reset, update user-facing canvas
   */
  useEffect(() => {
    if (shouldReloadCanvas) {
      canvas.loadFromJSON(currentState, () => {})
    }
  }, [canvas, shouldReloadCanvas, currentState])


  /**
   * Update canvas with selected tool
   * PS: add initialDrawing in dependency so user drawing update reinits tool
   */
  useEffect(() => {
    // Update canvas events with selected tool
    const selectedTool = new tools[drawingMode](canvas) as FabricTool
    const cleanupToolEvents = selectedTool.configureCanvas({
      fillColor: fillColor,
      strokeWidth: strokeWidth,
      strokeColor: strokeColor,
      displayRadius: displayRadius
    })

    /*
     * Ensure zoom/pan do not exceed the boundary of canvas.
     */
    let ensure_boundary: () => void = function (): void {
      const T = canvas.viewportTransform;

      if (img.aCoords == null || T == null) return

      const brRaw = img.aCoords.br
      const tlRaw = img.aCoords.tl

      const br = fabric.util.transformPoint(brRaw, T);
      const tl = fabric.util.transformPoint(tlRaw, T);

      const {
        x: left,
        y: top
      } = tl;

      const {
        x: right,
        y: bottom
      } = br;

      const width = canvas.getWidth()
      const height = canvas.getHeight()

      // calculate how far to translate to line up the edge of the object with  
      // the edge of the canvas                                                 
      const dLeft = Math.abs(right - width);
      const dRight = Math.abs(left);
      const dUp = Math.abs(bottom - height);
      const dDown = Math.abs(top);
      const maxDx = Math.min(dLeft, dRight);
      const maxDy = Math.min(dUp, dDown);

      // if the object is larger than the canvas, clamp translation such that   
      // we don't push the opposite boundary past the edge                      
      const leftIsOver = left < 0;
      const rightIsOver = right > width;
      const topIsOver = top < 0;
      const bottomIsOver = bottom > height;

      const translateLeft = rightIsOver && !leftIsOver;
      const translateRight = leftIsOver && !rightIsOver;
      const translateUp = bottomIsOver && !topIsOver;
      const translateDown = topIsOver && !bottomIsOver;

      const dx = translateLeft ? -maxDx : translateRight ? maxDx : 0;
      const dy = translateUp ? -maxDy : translateDown ? maxDy : 0;
      
      if (dx || dy) {
        T[4] += dx;
        T[5] += dy;
        canvas.requestRenderAll();

        backgroundCanvas.setViewportTransform(T)
        backgroundCanvas.requestRenderAll()
      }

    };

    /*
     * Mouse down event.
     * IF user press Alt keyboard, then move => Drag & Drop the image.
     */
    canvas.on("mouse:down", function (this: CustomFabricCanvas, opt) {
      var evt = opt.e as MouseEvent;

      if (evt.altKey === true) {
        this.isDragging = true;
        this.selection = false;
        this.lastPosX = evt.clientX;
        this.lastPosY = evt.clientY;

        canvas.setCursor('grab')
        // canvas.discardActiveObject();
        // canvas.requestRenderAll();

      }
      
      if (opt.target) {
        if (opt.target.type === 'rect') {

          const selectObject = canvas.getActiveObject()
          const selectIndex = canvas.getObjects().indexOf(selectObject)

          selectObject.selectionBackgroundColor = 'rgba(63,245,39,0.5)'

          // Return selected object.
          setSelectedRect(selectIndex)

          const data = canvas
              .getContext()
              .canvas.toDataURL()

          Streamlit.setComponentValue({
              data: data,
              width: canvas.getWidth(),
              height: canvas.getHeight(),
              raw: canvas.toObject(),
              selectIndex: selectIndex
          })

        }
      } else {
        setSelectedRect(-1)
      } 
    })

    
    /*
     * Mouse move event. Only affect while the alt key is pressed.
     */
    canvas.on("mouse:move", function (this: CustomFabricCanvas, opt) {    
      var e = opt.e as MouseEvent

      if (this.isDragging || false) {
        canvas.setCursor('grab')
        const delta = new fabric.Point( e.movementX, e.movementY )

        canvas.relativePan( delta )
        backgroundCanvas.relativePan( delta )

        ensure_boundary()

        e.preventDefault();
        e.stopPropagation();

      }
    })
       
    /*
     * Mouse wheel event - Scale in/out 
     */ 
    canvas.on("mouse:wheel", function (this: CustomFabricCanvas, opt) {
      var e = opt.e as WheelEvent;
      var delta = e.deltaY;
      var zoom = canvas.getZoom();
      zoom *= 0.999 ** delta;
      if (zoom > 10) zoom = 10;
      if (zoom < 0.1) zoom = 0.1;
      var point = new fabric.Point(e.offsetX, e.offsetY); 
      canvas.zoomToPoint(point, zoom);
      backgroundCanvas.zoomToPoint(point, zoom);

      e.preventDefault();
      e.stopPropagation();
    })

    canvas.on("mouse:up", (e: any) => {
      /*
       * There are several events can end with mouse:up:
       * 1. [rect] create new object
       * 2. [transform] resize selected object 
       * 3. [transform] choose selected object 
       * 4. [transform] delete selected object
       */

      // saveState(canvas.toJSON());
    
      var isEqualState = isEqual( canvas.toObject(), canvas.currentState )
      if ( (isEqualState === false) && (drawingMode === 'transform') ){
        canvas.currentState = { ...canvas.toObject() }

        const selectObject = canvas.getActiveObject()
        const selectIndex = canvas.getObjects().indexOf(selectObject)

        const data = canvas
            .getContext()
            .canvas.toDataURL()

        Streamlit.setComponentValue({
            data: data,
            width: canvas.getWidth(),
            height: canvas.getHeight(),
            raw: canvas.toObject(),
            selectIndex: selectIndex
        })

      }

      // Add your logic here for handling mouse up events
      canvas.isDragging = false;
      canvas.selection = true;
      canvas.setCursor("default")
    });

    canvas.on("mouse:dblclick", () => {
      if (drawingMode === 'transform') {
        const selectObject = canvas.getActiveObject()
        const selectIndex = canvas.getObjects().indexOf(selectObject)

        canvas.remove(selectObject)
        
        const data = canvas
          .getContext()
          .canvas.toDataURL()

        Streamlit.setComponentValue({
            data: data,
            width: canvas.getWidth(),
            height: canvas.getHeight(),
            raw: canvas.toObject(),
            selectIndex: selectIndex
        })

      }

    })

    // Cleanup tool + send data to Streamlit events
    return () => {
      cleanupToolEvents()
      canvas.off("mouse:down")
      canvas.off("mouse:move")
      canvas.off("mouse:up")
      canvas.off("mouse:wheel")
      canvas.off("mouse:dblclick")
      backgroundCanvas.off("mouse:down")
      backgroundCanvas.off("mouse:move")
      backgroundCanvas.off("mouse:up")
      backgroundCanvas.off("mouse:wheel")
      backgroundCanvas.off("mouse:dblclick")
    }
  }, [
    canvas,
    backgroundCanvas,
    strokeWidth,
    strokeColor,
    displayRadius,
    fillColor,
    drawingMode,
    initialDrawing,
    saveState,
    forceStreamlitUpdate,
    img
  ])

  /**
   * Render canvas w/ toolbar
   */
  return (
    <div style={{ position: "relative" }}>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          zIndex: -10,
          visibility: "hidden",
        }}
      >
        {/*<UpdateStreamlit
          canvasHeight={canvasHeight}
          canvasWidth={canvasWidth}
          shouldSendToStreamlit={
            realtimeUpdateStreamlit || forceSendToStreamlit
          }
          stateToSendToStreamlit={currentState}
          selectedRect={selectedRect}
        />*/}

      </div>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          zIndex: 0,
        }}
      >
        <canvas
          id="backgroundimage-canvas"
          width={canvasWidth}
          height={canvasHeight}
        />
      </div>
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          zIndex: 10,
        }}
      >
        <canvas
          id="canvas"
          width={canvasWidth}
          height={canvasHeight}
          style={{ border: "lightgrey 1px solid" }}
        />
      </div>
      {displayToolbar && (
        <CanvasToolbar
          topPosition={canvasHeight}
          leftPosition={canvasWidth}
          canUndo={canUndo}
          canRedo={canRedo}
          downloadCallback={forceStreamlitUpdate}
          undoCallback={undo}
          redoCallback={redo}
          resetCallback={() => {
            resetState(initialState)
          }}
        />
      )}
    </div>
  )
}

export default withStreamlitConnection(DrawableCanvas)