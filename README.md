# polyvore_vgg_visualization

This repository show visualized filter from vgg19_bn trained to classify polyvore dataset which have 380 categories. All categories are related to fashion items, accessories and so on.

## Convolutional Neural Network Filter Visualization

<table border=0 width="50px" >
	<tbody> 
		<tr>
			<td width="20%" align="center"> Layer 17 </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l17_f5_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l17_f10_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l17_f15_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l17_f50_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l17_f150_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l17_f200_iter100.jpg"> </td>
		</tr>
        <tr>
			<td width="20%" align="center"> Layer 27 </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l27_f5_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l27_f10_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l27_f100_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l27_f200_iter100.jpg"> </td>
		</tr>
        <tr>
			<td width="20%" align="center"> Layer 36 </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l36_f5_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l36_f10_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l36_f100_iter100.jpg"> </td>
			<td width="13.3%" align="center"> <img src="images/layer_vis_l36_f200_iter100.jpg"> </td>
		</tr>
	</tbody>
</table>

## Reference
1. pytorch_cnn_visualization : [link](https://github.com/Jungjaewon/pytorch-cnn-visualizations)
2. polyvore_dataset : [link](https://github.com/xthan/polyvore)