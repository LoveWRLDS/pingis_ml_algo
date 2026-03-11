package com.app.skatingbergs

import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.util.AttributeSet
import android.view.ViewGroup
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

@Composable
fun ShoeViewport(
    modifier: Modifier = Modifier,
    orientation: Quaternionf
) {
    AndroidView(
        modifier = modifier,
        factory = { context ->
            ShoeSurfaceView(context).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            }
        },
        update = { view ->
            view.setOrientation(orientation)
        }
    )
}

private class ShoeSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {
    private val renderer = ShoeRenderer()

    init {
        setEGLContextClientVersion(2)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun setOrientation(orientation: Quaternionf) {
        renderer.setOrientation(orientation)
    }
}

private class ShoeRenderer : GLSurfaceView.Renderer {
    private val viewMatrix = FloatArray(16)
    private val projectionMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    private val mvpMatrix = FloatArray(16)
    private val tempMatrix = FloatArray(16)
    private val rotationMatrix = FloatArray(16)

    @Volatile
    private var orientation = Quaternionf.Identity

    private var program = 0
    private var positionHandle = 0
    private var mvpHandle = 0
    private var colorHandle = 0

    private lateinit var cubeVertices: FloatBuffer
    private lateinit var cubeIndices: ShortBuffer

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.96f, 0.97f, 0.99f, 1f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)

        cubeVertices = floatBufferOf(
            -0.5f, -0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            0.5f, 0.5f, 0.5f,
            -0.5f, 0.5f, 0.5f,
            -0.5f, -0.5f, -0.5f,
            0.5f, -0.5f, -0.5f,
            0.5f, 0.5f, -0.5f,
            -0.5f, 0.5f, -0.5f
        )
        cubeIndices = shortBufferOf(
            0, 1, 2, 0, 2, 3,
            1, 5, 6, 1, 6, 2,
            5, 4, 7, 5, 7, 6,
            4, 0, 3, 4, 3, 7,
            3, 2, 6, 3, 6, 7,
            4, 5, 1, 4, 1, 0
        )

        program = buildProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        positionHandle = GLES20.glGetAttribLocation(program, "aPosition")
        mvpHandle = GLES20.glGetUniformLocation(program, "uMvpMatrix")
        colorHandle = GLES20.glGetUniformLocation(program, "uColor")

        Matrix.setLookAtM(
            viewMatrix,
            0,
            0f,
            1.1f,
            4.4f,
            0f,
            0.5f,
            0f,
            0f,
            1f,
            0f
        )
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        val ratio = width.toFloat() / height.coerceAtLeast(1).toFloat()
        Matrix.perspectiveM(projectionMatrix, 0, 45f, ratio, 0.1f, 100f)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        GLES20.glUseProgram(program)

        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 3 * 4, cubeVertices)

        val orientationMatrix = orientation.toMatrix4()
        System.arraycopy(orientationMatrix, 0, rotationMatrix, 0, orientationMatrix.size)

        drawPart(
            translation = floatArrayOf(0f, 0.55f, 0f),
            scale = floatArrayOf(0.9f, 1.2f, 0.45f),
            color = floatArrayOf(0.85f, 0.92f, 1.0f, 1f)
        )
        drawPart(
            translation = floatArrayOf(0.38f, 0.2f, 0f),
            scale = floatArrayOf(0.95f, 0.45f, 0.5f),
            color = floatArrayOf(0.72f, 0.85f, 0.98f, 1f)
        )
        drawPart(
            translation = floatArrayOf(-0.2f, -0.05f, 0f),
            scale = floatArrayOf(0.32f, 0.28f, 0.42f),
            color = floatArrayOf(0.62f, 0.76f, 0.92f, 1f)
        )
        drawPart(
            translation = floatArrayOf(0.2f, -0.52f, 0f),
            scale = floatArrayOf(1.5f, 0.06f, 0.12f),
            color = floatArrayOf(0.65f, 0.68f, 0.73f, 1f)
        )
        drawPart(
            translation = floatArrayOf(-0.5f, -0.5f, 0f),
            scale = floatArrayOf(0.1f, 0.2f, 0.1f),
            color = floatArrayOf(0.45f, 0.48f, 0.55f, 1f)
        )

        GLES20.glDisableVertexAttribArray(positionHandle)
    }

    fun setOrientation(newOrientation: Quaternionf) {
        orientation = newOrientation.normalized()
    }

    private fun drawPart(
        translation: FloatArray,
        scale: FloatArray,
        color: FloatArray
    ) {
        Matrix.setIdentityM(modelMatrix, 0)
        Matrix.multiplyMM(tempMatrix, 0, rotationMatrix, 0, modelMatrix, 0)
        System.arraycopy(tempMatrix, 0, modelMatrix, 0, tempMatrix.size)
        Matrix.translateM(modelMatrix, 0, translation[0], translation[1], translation[2])
        Matrix.scaleM(modelMatrix, 0, scale[0], scale[1], scale[2])

        Matrix.multiplyMM(tempMatrix, 0, viewMatrix, 0, modelMatrix, 0)
        Matrix.multiplyMM(mvpMatrix, 0, projectionMatrix, 0, tempMatrix, 0)

        GLES20.glUniformMatrix4fv(mvpHandle, 1, false, mvpMatrix, 0)
        GLES20.glUniform4fv(colorHandle, 1, color, 0)
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, 36, GLES20.GL_UNSIGNED_SHORT, cubeIndices)
    }

    private fun buildProgram(vertexShaderCode: String, fragmentShaderCode: String): Int {
        val vertexShader = compileShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode)
        val fragmentShader = compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode)
        return GLES20.glCreateProgram().also { programId ->
            GLES20.glAttachShader(programId, vertexShader)
            GLES20.glAttachShader(programId, fragmentShader)
            GLES20.glLinkProgram(programId)
        }
    }

    private fun compileShader(type: Int, shaderCode: String): Int =
        GLES20.glCreateShader(type).also { shader ->
            GLES20.glShaderSource(shader, shaderCode)
            GLES20.glCompileShader(shader)
        }

    private fun floatBufferOf(vararg values: Float): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(values)
                position(0)
            }

    private fun shortBufferOf(vararg values: Short): ShortBuffer =
        ByteBuffer.allocateDirect(values.size * 2)
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()
            .apply {
                put(values)
                position(0)
            }

    companion object {
        private const val VERTEX_SHADER = """
            uniform mat4 uMvpMatrix;
            attribute vec4 aPosition;
            void main() {
                gl_Position = uMvpMatrix * aPosition;
            }
        """

        private const val FRAGMENT_SHADER = """
            precision mediump float;
            uniform vec4 uColor;
            void main() {
                gl_FragColor = uColor;
            }
        """
    }
}
