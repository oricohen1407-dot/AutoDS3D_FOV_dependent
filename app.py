import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from skimage import io
import gradio as gr
from func_utils import func1, func2, func3, func4, func5, func6_1, func6_2, func7

def create_output_window():
    return gr.Textbox(label="Output (response to button click)")


def create_description_section():
    with gr.Accordion('Description', open=False):
        gr.Markdown(
            """
            **An automated and enhanced version of DeepSTORM3D (DS3D) for PSF-engineering-based 3D localization microscopy**
            
            **Inputs**
            - Optical parameters  
            - PSF z-stack  
            - Image sequence (from a blinking video)
            
            **Outputs**
            - Localization list  
            - Intermediate results (figures)
            
            **Operation Modes**
            - Step by step  
            - One click
            """
        )


def create_optical_parameter_inputs():
    with gr.Accordion("parameter column 1", open=False):
        M = gr.Number(label="★objective magnification M", value=100)
        NA = gr.Number(label="★numeric aperture NA", value=1.40)
        n_immersion = gr.Number(label="★refractive index of immersion medium", value=1.518)
        lamda = gr.Number(label="emission wavelength lamda [um]", value=0.64)
        n_sample = gr.Number(label="★refractive index of sample", value=1.33)
        ps_camera = gr.Number(label="★camera pixel size at image plane [um]", value=13.8)

        f_4f = gr.Number(label="focal length of the 4f setup [um]", value=100e3, visible=False)
        ps_BFP = gr.Number(label="■ pixel size of mask plane [um]", value=30, visible=False)
        external_mask = gr.Textbox(label="external mask", value='None', visible=False)

    return [M, NA, n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask]


def create_calibration_parameter_inputs():
    with gr.Accordion('parameter column 2', open=False):
        zstack_file = gr.Textbox(label='★z-stack file (within app folder)', value=r'.\dataset3\calibration 1.tif')
        nfp_text = gr.Textbox(label="★z-stack NFPs (start, end, number) [um]", value='-0.5, 0.5, 41')

        NFP = gr.Number(label='★NFP [um]', value=0.5)
        zrange = gr.Textbox(label='★expected z range (z_min, z_max) [um]', value='0, 1.0')
        raw_image_folder = gr.Textbox(label='★image folder (within app folder)', value=r'.\dataset3\ims')
        snr_roi = gr.Textbox(label='★SNR detection ROI (r0, c0, r1, c1)', value='0, 0, 20, 20')
        max_pv = gr.Number(label='■ maximum pixel value (MPV)', value=250)
        projection_01 = gr.Number(label='0-1 projection', visible=False)

    return [zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01]


def create_blinking_parameter_inputs():
    with gr.Accordion('parameter column 3', open=False):
        num_z_voxel = gr.Number(label='★number of voxels in z', value=61)
        training_im_size = gr.Number(label='training image size', value=61)
        us_factor = gr.Number(label='★up-sampling factor (options: 1, 2, 4)', value=1)
        max_num_particles = gr.Number(label='maximum number of particles', value=20)
        num_training_images = gr.Number(label='number of training images', value=10000, visible=False)
        previous_param_dict = gr.Textbox(label="external training file", value='None')
        test_idx = gr.Number(label='■ test image index', value=10)
        threshold = gr.Number(label='■ threshold (0-1000)', value=40)

    return [num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold]


def create_action_buttons(input_all, output_window):
    with gr.Row():
        button1 = gr.Button("characterize PSF")
        button1.click(func1, inputs=input_all, outputs=output_window)

        button2 = gr.Button("preprocess images")
        button2.click(func2, inputs=input_all, outputs=output_window)

        button3 = gr.Button("characterize SNR")
        button3.click(func3, inputs=input_all, outputs=output_window)

    with gr.Row():
        button4 = gr.Button("simulate training data")
        button4.click(func4, inputs=input_all, outputs=output_window)

        button5 = gr.Button("train a localization neural net")
        button5.click(func5, inputs=input_all, outputs=output_window)

    with gr.Row():
        button6_0 = gr.Button("test the net")
        button6_0.click(func6_1, inputs=input_all, outputs=output_window)

        button6_1 = gr.Button("localize all")
        button6_1.click(func6_2, inputs=input_all, outputs=output_window)

    button7 = gr.Button("ONE CLICK", variant='huggingface')
    button7.click(func7, inputs=input_all, outputs=output_window)

    output_window.render()


def build_demo():
    output_window = create_output_window()

    with gr.Blocks() as demo:
        gr.Markdown("""
        # AutoDS3D
        """)
        create_description_section()

        state = gr.State(value={})

        with gr.Row():
            optical_inputs = create_optical_parameter_inputs()
            calibration_inputs = create_calibration_parameter_inputs()
            blinking_inputs = create_blinking_parameter_inputs()

        input_all = optical_inputs + calibration_inputs + blinking_inputs + [state]
        create_action_buttons(input_all, output_window)

    return demo


def choose_file():
    """Open a file dialog and let the user choose a file."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(title="Select a file")

    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    else:
        print("No file selected.")
        return None


def onselect(eclick, erelease):
    """
    Callback function for RectangleSelector.
    eclick  : MouseEvent at start of selection (mouse press)
    erelease: MouseEvent at end of selection (mouse release)
    """
    global roi_coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)  # Top-left corner (c0, r0)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)  # Bottom-right corner (c1, r1)
    roi_coords = [y1, x1, y2, x2]  # Store as (r0, c0, r1, c1)
    print(
        f'SNR detection ROI (r0, c0, r1, c1): {roi_coords[0]}, {roi_coords[1]}, {roi_coords[2]}, {roi_coords[3]}'
    )


if __name__ == "__main__":
    # file_path = choose_file()
    # if file_path is not None:
    #     image = io.imread(file_path)
    #     fig, ax = plt.subplots()
    #     ax.imshow(image)  # Display the image
    #     roi_coords = []  # Store ROI coordinates
    #     rect_selector = RectangleSelector(ax, onselect, interactive=True, button=[1])
    #     plt.show()

    demo = build_demo()
    demo.launch(share=False)  # local computer
    # demo.launch(share=False, server_name="132.68.109.79")  # remote server





