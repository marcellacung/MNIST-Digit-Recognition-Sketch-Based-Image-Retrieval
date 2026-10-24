# network_visualizer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import numpy as np
import os
from pathlib import Path
import json

class NetworkVisualizer:
    def __init__(self, output_dir='static/network_diagrams'):
        """Initialize the visualizer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define colors for different layer types
        self.colors = {
            'input': '#4CAF50',      # Green
            'conv': '#2196F3',       # Blue
            'pool': '#FF9800',       # Orange
            'fc': '#9C27B0',         # Purple
            'output': '#F44336',     # Red
            'flatten': '#795548',    # Brown
            'dropout': '#607D8B',    # Blue Grey
            'activation': '#00BCD4', # Cyan
        }
        
        # Define layer descriptions
        self.layer_descriptions = {
            'conv': "Convolutional Layer\n• Applies filters to extract features\n• Learns spatial hierarchies\n• Parameters: kernel_size × filters × input_channels",
            'pool': "Pooling Layer\n• Reduces spatial dimensions\n• Max or Average pooling\n• Provides translation invariance",
            'fc': "Fully Connected Layer\n• Every neuron connects to all inputs\n• Learns global patterns\n• High parameter count",
            'flatten': "Flatten Layer\n• Converts 2D/3D features to 1D\n• Prepares for FC layers\n• No learnable parameters",
            'dropout': "Dropout Layer\n• Regularization technique\n• Randomly drops neurons\n• Prevents overfitting",
            'activation': "Activation Function\n• Adds non-linearity\n• ReLU, Sigmoid, Tanh\n• No learnable parameters"
        }
    
    def create_fc_network_diagram(self):
        """Create diagram for Fully Connected Network"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Layer visualization
        self._draw_fc_layers(ax1)
        
        # Right: Detailed architecture
        self._draw_fc_architecture_details(ax2)
        
        plt.suptitle('Fully Connected Network Architecture', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save diagram
        output_path = os.path.join(self.output_dir, 'fc_network.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ FC Network diagram saved to {output_path}")
        
        # Create text description
        fc_description = self._create_fc_description()
        return output_path, fc_description
    
    def create_cnn_network_diagram(self):
        """Create diagram for Convolutional Neural Network"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        
        # Left: Layer visualization with feature maps
        self._draw_cnn_layers(ax1)
        
        # Right: Architecture details and feature visualization
        self._draw_cnn_architecture_details(ax2)
        
        plt.suptitle('Convolutional Neural Network Architecture', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save diagram
        output_path = os.path.join(self.output_dir, 'cnn_network.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ CNN Network diagram saved to {output_path}")
        
        # Create text description
        cnn_description = self._create_cnn_description()
        return output_path, cnn_description
    
    def create_comparison_diagram(self):
        """Create side-by-side comparison of FC and CNN"""
        fig = plt.figure(figsize=(20, 10))
        
        # FC Network
        ax1 = plt.subplot(1, 2, 1)
        self._draw_fc_layers_simple(ax1)
        ax1.set_title('Fully Connected Network', fontsize=14, fontweight='bold', pad=20)
        
        # CNN Network
        ax2 = plt.subplot(1, 2, 2)
        self._draw_cnn_layers_simple(ax2)
        ax2.set_title('Convolutional Neural Network', fontsize=14, fontweight='bold', pad=20)
        
        # Add comparison text
        comparison_text = """Comparison:
• FC: Dense connections, global features
• CNN: Sparse connections, local features
• FC: High parameters, prone to overfitting
• CNN: Parameter efficient, translation invariant
• FC: Input size fixed
• CNN: Handles variable input sizes
• FC: No spatial hierarchy
• CNN: Learns spatial hierarchies"""
        
        plt.figtext(0.5, 0.02, comparison_text, ha='center', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('CNN vs Fully Connected Network Comparison', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        # Save diagram
        output_path = os.path.join(self.output_dir, 'comparison_diagram.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison diagram saved to {output_path}")
        return output_path
    
    def create_layer_wise_visualization(self):
        """Create visualization showing feature transformations"""
        fig = plt.figure(figsize=(15, 10))
        
        # CNN Feature Progression
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # Original input
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_mnist_digit(ax1, digit=7)
        ax1.set_title('Input: Digit 7\n28×28×1', fontsize=10)
        
        # After Conv1
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_feature_maps(ax2, maps=32, title='Conv1 Output\n28×28×32')
        
        # After Pool1
        ax3 = fig.add_subplot(gs[0, 2])
        self._draw_feature_maps(ax3, maps=32, size=14, title='Pool1 Output\n14×14×32')
        
        # After Conv2
        ax4 = fig.add_subplot(gs[0, 3])
        self._draw_feature_maps(ax4, maps=64, size=14, title='Conv2 Output\n14×14×64')
        
        # After Pool2
        ax5 = fig.add_subplot(gs[1, 0])
        self._draw_feature_maps(ax5, maps=64, size=7, title='Pool2 Output\n7×7×64')
        
        # Flattened
        ax6 = fig.add_subplot(gs[1, 1])
        self._draw_flattened_features(ax6, count=3136, title='Flattened\n3136 neurons')
        
        # FC Layers
        ax7 = fig.add_subplot(gs[1, 2])
        self._draw_fc_layer(ax7, neurons=128, title='FC1\n128 neurons')
        
        # Output
        ax8 = fig.add_subplot(gs[1, 3])
        self._draw_output_layer(ax8, title='Output\n10 classes')
        
        plt.suptitle('CNN Feature Transformation Process', fontsize=14, fontweight='bold', y=0.98)
        
        output_path = os.path.join(self.output_dir, 'feature_transformation.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature transformation diagram saved to {output_path}")
        return output_path
    
    def _draw_fc_layers(self, ax):
        """Draw Fully Connected layers"""
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 6)
        ax.axis('off')
        
        # Layer definitions: (x_pos, neuron_count, label, color)
        layers = [
            (1, 784, 'Input\n784 neurons', 'input'),
            (3, 256, 'Hidden 1\n256 neurons', 'fc'),
            (5, 128, 'Hidden 2\n128 neurons', 'fc'),
            (7, 64, 'Hidden 3\n64 neurons', 'fc'),
            (9, 10, 'Output\n10 classes', 'output')
        ]
        
        # Draw layers
        for x, neurons, label, layer_type in layers:
            # Draw layer box
            box_height = min(4, neurons / 200)  # Scale height
            rect = patches.Rectangle((x-0.3, -box_height/2), 0.6, box_height,
                                   facecolor=self.colors[layer_type], alpha=0.7,
                                   edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            ax.text(x, -box_height/2 - 0.3, label, ha='center', va='top',
                   fontsize=9, fontweight='bold')
            
            # Draw neurons as circles
            if neurons <= 100:  # Only draw circles for reasonable counts
                for i in range(min(neurons, 20)):  # Show up to 20 neurons
                    y = -box_height/2 + box_height * (i+1)/(min(neurons, 20)+1)
                    circle = Circle((x, y), 0.03, color='black', alpha=0.5)
                    ax.add_patch(circle)
            else:
                # Show neuron count
                ax.text(x, 0, f"{neurons}\nneurons", ha='center', va='center',
                       fontsize=8, fontweight='bold')
        
        # Draw connections between layers
        for i in range(len(layers)-1):
            x1 = layers[i][0] + 0.3
            x2 = layers[i+1][0] - 0.3
            
            # Draw multiple connections
            for y in np.linspace(-1, 1, 5):
                arrow = FancyArrowPatch((x1, y), (x2, y),
                                      arrowstyle='->', color='gray',
                                      alpha=0.3, mutation_scale=10)
                ax.add_patch(arrow)
        
        # Add title and info
        ax.text(5, 4.5, 'Fully Connected Network', ha='center', fontsize=12, fontweight='bold')
        
        info_text = """Characteristics:
• Every neuron connects to all neurons in next layer
• High parameter count (易过拟合)
• No spatial information preservation
• Simple but computationally expensive"""
        
        ax.text(5, 3.5, info_text, ha='center', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
    
    def _draw_fc_architecture_details(self, ax):
        """Draw FC architecture details table"""
        ax.axis('off')
        
        # Architecture table
        layers_data = [
            ('Layer', 'Type', 'Output Shape', 'Parameters', 'Activation'),
            ('Input', 'Input', '(None, 784)', '0', '-'),
            ('Dense1', 'Fully Connected', '(None, 256)', '200,960', 'ReLU'),
            ('Dense2', 'Fully Connected', '(None, 128)', '32,896', 'ReLU'),
            ('Dense3', 'Fully Connected', '(None, 64)', '8,256', 'ReLU'),
            ('Output', 'Dense', '(None, 10)', '650', 'Softmax')
        ]
        
        # Calculate positions
        y_positions = np.linspace(5, 0, len(layers_data))
        
        # Draw table
        for i, (layer, ltype, shape, params, activation) in enumerate(layers_data):
            y = y_positions[i]
            
            # Header row
            if i == 0:
                ax.text(2, y, layer, ha='center', fontweight='bold', fontsize=10)
                ax.text(4, y, ltype, ha='center', fontweight='bold', fontsize=10)
                ax.text(6, y, shape, ha='center', fontweight='bold', fontsize=10)
                ax.text(8, y, params, ha='center', fontweight='bold', fontsize=10)
                ax.text(10, y, activation, ha='center', fontweight='bold', fontsize=10)
                
                # Header line
                ax.plot([1, 11], [y-0.2, y-0.2], 'k-', linewidth=2)
            else:
                # Data rows
                color = self.colors['fc'] if 'Dense' in layer else self.colors['output'] if 'Output' in layer else '#FFFFFF'
                
                # Draw row background
                rect = patches.Rectangle((1, y-0.4), 10, 0.8,
                                       facecolor=color, alpha=0.2,
                                       edgecolor='gray')
                ax.add_patch(rect)
                
                ax.text(2, y, layer, ha='center', fontsize=9)
                ax.text(4, y, ltype, ha='center', fontsize=9)
                ax.text(6, y, shape, ha='center', fontsize=9)
                ax.text(8, y, params, ha='center', fontsize=9)
                ax.text(10, y, activation, ha='center', fontsize=9)
        
        # Add parameter summary
        total_params = 200960 + 32896 + 8256 + 650
        summary_text = f"""Total Parameters: {total_params:,}
Trainable: {total_params:,}
Non-trainable: 0
Model Size: ~{total_params * 4 / 1024:.1f} KB"""
        
        ax.text(6, -1.5, summary_text, ha='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.5))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(-2, 6)
    
    def _draw_cnn_layers(self, ax):
        """Draw CNN layers with feature maps"""
        ax.set_xlim(-1, 13)
        ax.set_ylim(-2, 6)
        ax.axis('off')
        
        # Layer sequence
        layers = [
            (1, 'Input\n28×28×1', 'input', 28),
            (3, 'Conv1\n28×28×32', 'conv', 14),
            (5, 'Pool1\n14×14×32', 'pool', 8),
            (7, 'Conv2\n14×14×64', 'conv', 6),
            (9, 'Pool2\n7×7×64', 'pool', 4),
            (11, 'Output\n10', 'output', 2)
        ]
        
        # Draw layers
        for x, label, layer_type, size in layers:
            # Draw feature map representation
            if layer_type in ['conv', 'pool']:
                # Draw multiple feature maps
                for i in range(3):  # Show 3 sample feature maps
                    y_offset = i - 1
                    rect = patches.Rectangle((x-0.5, y_offset-0.3), 1, 0.6,
                                           facecolor=self.colors[layer_type],
                                           alpha=0.7 - i*0.2,
                                           edgecolor='black')
                    ax.add_patch(rect)
                    
                    # Add grid inside feature map
                    if size > 4:
                        for grid_x in np.linspace(x-0.5, x+0.5, 4):
                            ax.plot([grid_x, grid_x], [y_offset-0.3, y_offset+0.3], 
                                   'gray', alpha=0.2, linewidth=0.5)
                        for grid_y in np.linspace(y_offset-0.3, y_offset+0.3, 4):
                            ax.plot([x-0.5, x+0.5], [grid_y, grid_y], 
                                   'gray', alpha=0.2, linewidth=0.5)
            else:
                # Draw single box for input/output
                rect = patches.Rectangle((x-0.5, -0.5), 1, 1,
                                       facecolor=self.colors[layer_type], alpha=0.7,
                                       edgecolor='black', linewidth=2)
                ax.add_patch(rect)
            
            # Add label
            ax.text(x, -1.8, label, ha='center', va='top',
                   fontsize=9, fontweight='bold')
        
        # Draw connections
        for i in range(len(layers)-1):
            x1 = layers[i][0] + 0.5
            x2 = layers[i+1][0] - 0.5
            
            # Draw arrows
            arrow = FancyArrowPatch((x1, 0), (x2, 0),
                                  arrowstyle='->', color='black',
                                  alpha=0.7, mutation_scale=15, linewidth=1.5)
            ax.add_patch(arrow)
            
            # Add connection label
            if i == 1:
                ax.text((x1+x2)/2, 0.5, 'Downsampling', ha='center',
                       fontsize=8, style='italic')
            elif i == 3:
                ax.text((x1+x2)/2, 0.5, 'Downsampling', ha='center',
                       fontsize=8, style='italic')
        
        # Add title and info
        ax.text(6, 5.5, 'Convolutional Neural Network', ha='center', fontsize=12, fontweight='bold')
        
        info_text = """Characteristics:
• Local receptive fields (局部感受野)
• Weight sharing (参数共享)
• Spatial hierarchy (空间层次结构)
• Translation invariance (平移不变性)
• Parameter efficient (参数高效)"""
        
        ax.text(6, 4.2, info_text, ha='center', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.3))
        
        # Add convolution visualization
        ax.text(6, 2.8, "Convolution Operation:", ha='center', fontsize=10, fontweight='bold')
        self._draw_convolution_example(ax, x=6, y=2.3)
    
    def _draw_convolution_example(self, ax, x, y):
        """Draw a small convolution visualization"""
        # Input grid
        input_size = 3
        for i in range(input_size):
            for j in range(input_size):
                rect = Rectangle((x-0.8+i*0.3, y-0.4+j*0.3), 0.25, 0.25,
                               facecolor='lightblue', edgecolor='black')
                ax.add_patch(rect)
        
        # Kernel
        kernel_size = 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                rect = Rectangle((x+0.1+i*0.3, y-0.1+j*0.3), 0.25, 0.25,
                               facecolor='orange', edgecolor='black')
                ax.add_patch(rect)
        
        # Arrow
        ax.annotate('', xy=(x+0.8, y+0.4), xytext=(x-0.1, y+0.4),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        
        # Output
        output_size = 2
        for i in range(output_size):
            for j in range(output_size):
                rect = Rectangle((x+1.2+i*0.3, y+0.1+j*0.3), 0.25, 0.25,
                               facecolor='lightgreen', edgecolor='black')
                ax.add_patch(rect)
        
        # Labels
        ax.text(x, y+0.8, "Input", ha='center', fontsize=8)
        ax.text(x+0.55, y+0.8, "Kernel", ha='center', fontsize=8)
        ax.text(x+1.35, y+0.8, "Output", ha='center', fontsize=8)
    
    def _draw_cnn_architecture_details(self, ax):
        """Draw CNN architecture details"""
        ax.axis('off')
        
        # Architecture table
        layers_data = [
            ('Layer', 'Type', 'Output Shape', 'Parameters', 'Kernel Size'),
            ('Input', 'Input', '(None, 28, 28, 1)', '0', '-'),
            ('Conv2D_1', 'Conv2D', '(None, 28, 28, 32)', '320', '3×3'),
            ('MaxPool2D_1', 'MaxPooling', '(None, 14, 14, 32)', '0', '2×2'),
            ('Conv2D_2', 'Conv2D', '(None, 14, 14, 64)', '18,496', '3×3'),
            ('MaxPool2D_2', 'MaxPooling', '(None, 7, 7, 64)', '0', '2×2'),
            ('Flatten', 'Flatten', '(None, 3136)', '0', '-'),
            ('Dense_1', 'Dense', '(None, 128)', '401,536', '-'),
            ('Dense_2', 'Dense', '(None, 10)', '1,290', '-')
        ]
        
        # Calculate positions
        y_positions = np.linspace(5, -2, len(layers_data))
        
        # Draw table
        for i, (layer, ltype, shape, params, kernel) in enumerate(layers_data):
            y = y_positions[i]
            
            # Header row
            if i == 0:
                ax.text(2, y, layer, ha='center', fontweight='bold', fontsize=10)
                ax.text(4, y, ltype, ha='center', fontweight='bold', fontsize=10)
                ax.text(6, y, shape, ha='center', fontweight='bold', fontsize=10)
                ax.text(8, y, params, ha='center', fontweight='bold', fontsize=10)
                ax.text(10, y, kernel, ha='center', fontweight='bold', fontsize=10)
                
                # Header line
                ax.plot([1, 11], [y-0.2, y-0.2], 'k-', linewidth=2)
            else:
                # Determine color based on layer type
                if 'Conv' in layer:
                    color = self.colors['conv']
                elif 'Pool' in layer:
                    color = self.colors['pool']
                elif 'Flatten' in layer:
                    color = self.colors['flatten']
                elif 'Dense' in layer:
                    color = self.colors['fc']
                else:
                    color = '#FFFFFF'
                
                # Draw row background
                rect = patches.Rectangle((1, y-0.4), 10, 0.8,
                                       facecolor=color, alpha=0.2,
                                       edgecolor='gray')
                ax.add_patch(rect)
                
                ax.text(2, y, layer, ha='center', fontsize=9)
                ax.text(4, y, ltype, ha='center', fontsize=9)
                ax.text(6, y, shape, ha='center', fontsize=9)
                ax.text(8, y, params, ha='center', fontsize=9)
                ax.text(10, y, kernel, ha='center', fontsize=9)
        
        # Add parameter summary
        total_params = 320 + 18496 + 401536 + 1290
        summary_text = f"""Total Parameters: {total_params:,}
Trainable: {total_params:,}
Non-trainable: 0
Model Size: ~{total_params * 4 / 1024:.1f} KB
Parameter Efficiency: ~{total_params/242_000:.1%} of equivalent FC"""
        
        ax.text(6, -3.5, summary_text, ha='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.5))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(-4, 6)
    
    def _draw_fc_layers_simple(self, ax):
        """Simple FC layer visualization"""
        ax.axis('off')
        ax.set_xlim(-1, 7)
        ax.set_ylim(-1, 5)
        
        # Draw layers
        for i, (x, neurons, label) in enumerate([(1, 784, '784'), (3, 256, '256'), 
                                                 (5, 128, '128'), (7, 10, '10')]):
            # Layer circle
            circle = Circle((x, 2), 0.8, facecolor=self.colors['fc' if i < 3 else 'output'],
                          alpha=0.7, edgecolor='black')
            ax.add_patch(circle)
            ax.text(x, 2, label, ha='center', va='center', fontweight='bold')
            
            # Layer label
            layer_names = ['Input', 'Hidden1', 'Hidden2', 'Output']
            ax.text(x, 0.2, layer_names[i], ha='center', fontsize=9)
            
            # Draw connections
            if i > 0:
                ax.plot([x-1.8, x-0.8], [2, 2], 'gray', alpha=0.3, linewidth=1)
        
        ax.text(4, 4.5, 'Fully Connected', ha='center', fontweight='bold', fontsize=11)
    
    def _draw_cnn_layers_simple(self, ax):
        """Simple CNN layer visualization"""
        ax.axis('off')
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 5)
        
        # Draw layers
        layers = [
            (1, '28×28', 'input'),
            (3, 'Conv\n32', 'conv'),
            (5, 'Pool\n14×14', 'pool'),
            (7, '10', 'output')
        ]
        
        for x, label, layer_type in layers:
            if layer_type in ['conv', 'pool']:
                # Feature map representation
                rect = Rectangle((x-0.6, 1.2), 1.2, 1.6,
                               facecolor=self.colors[layer_type], alpha=0.7,
                               edgecolor='black')
                ax.add_patch(rect)
                
                # Add grid lines
                for i in range(3):
                    ax.plot([x-0.6, x+0.6], [1.2 + i*0.4, 1.2 + i*0.4], 
                           'gray', alpha=0.3, linewidth=0.5)
                    ax.plot([x-0.6 + i*0.6, x-0.6 + i*0.6], [1.2, 2.8], 
                           'gray', alpha=0.3, linewidth=0.5)
            else:
                # Circle for input/output
                circle = Circle((x, 2), 0.6, facecolor=self.colors[layer_type],
                              alpha=0.7, edgecolor='black')
                ax.add_patch(circle)
            
            ax.text(x, 2, label, ha='center', va='center', fontweight='bold', fontsize=9)
            
            # Layer name
            layer_names = ['Input', 'Conv', 'Pool', 'Output']
            ax.text(x, 0.2, layer_names[layers.index((x, label, layer_type))], 
                   ha='center', fontsize=9)
            
            # Draw connections
            if x > 1:
                ax.plot([x-1.8, x-0.6], [2, 2], 'black', alpha=0.5, linewidth=1)
        
        # Add convolution illustration
        self._draw_mini_conv(ax, x=8, y=3.5)
        
        ax.text(4, 4.5, 'Convolutional', ha='center', fontweight='bold', fontsize=11)
    
    def _draw_mini_conv(self, ax, x, y):
        """Draw mini convolution illustration"""
        # Input 3x3
        for i in range(3):
            for j in range(3):
                rect = Rectangle((x+i*0.3, y+j*0.3), 0.25, 0.25,
                               facecolor='lightblue', edgecolor='gray')
                ax.add_patch(rect)
        
        # Kernel 2x2
        for i in range(2):
            for j in range(2):
                rect = Rectangle((x+1.2+i*0.3, y+0.3+j*0.3), 0.25, 0.25,
                               facecolor='orange', edgecolor='gray')
                ax.add_patch(rect)
        
        # Output 2x2
        for i in range(2):
            for j in range(2):
                rect = Rectangle((x+2.4+i*0.3, y+0.3+j*0.3), 0.25, 0.25,
                               facecolor='lightgreen', edgecolor='gray')
                ax.add_patch(rect)
        
        # Arrows
        ax.annotate('', xy=(x+1.1, y+0.8), xytext=(x+0.9, y+0.8),
                   arrowprops=dict(arrowstyle='->', lw=1))
        ax.annotate('', xy=(x+2.3, y+0.8), xytext=(x+2.1, y+0.8),
                   arrowprops=dict(arrowstyle='->', lw=1))
        
        ax.text(x+0.45, y+1.2, 'Input', ha='center', fontsize=7)
        ax.text(x+1.65, y+1.2, 'Kernel', ha='center', fontsize=7)
        ax.text(x+2.85, y+1.2, 'Output', ha='center', fontsize=7)
    
    def _draw_mnist_digit(self, ax, digit=7):
        """Draw an MNIST digit"""
        ax.axis('off')
        ax.set_xlim(0, 28)
        ax.set_ylim(0, 28)
        
        # Create a simple 7 shape
        points = [(5, 23), (23, 23), (8, 5), (5, 8)]
        
        # Draw the digit
        for i in range(len(points)-1):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            ax.plot([x1, x2], [28-y1, 28-y2], 'k-', linewidth=3)
        
        ax.fill_between([5, 23], [28-23, 28-23], [28-20, 28-20], color='black', alpha=0.1)
        
        # Add grid
        for i in range(0, 29, 4):
            ax.plot([i, i], [0, 28], 'gray', alpha=0.1, linewidth=0.5)
            ax.plot([0, 28], [i, i], 'gray', alpha=0.1, linewidth=0.5)
    
    def _draw_feature_maps(self, ax, maps=32, size=28, title=''):
        """Draw feature maps representation"""
        ax.axis('off')
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        
        # Draw 4 sample feature maps (3x3 grid)
        for i in range(2):
            for j in range(2):
                rect = Rectangle((i, j), 1, 1,
                               facecolor=self.colors['conv'], alpha=0.7,
                               edgecolor='black')
                ax.add_patch(rect)
                
                # Add some random activation pattern
                for x in np.linspace(i+0.1, i+0.9, 3):
                    for y in np.linspace(j+0.1, j+0.9, 3):
                        intensity = np.random.random() * 0.8 + 0.2
                        circle = Circle((x, y), 0.1,
                                      facecolor=str(intensity),  # Grayscale
                                      edgecolor='none')
                        ax.add_patch(circle)
        
        ax.text(1.5, -0.2, f"{title}", ha='center', fontsize=8)
        ax.text(1.5, 3.2, f"×{maps//4} maps", ha='center', fontsize=7, style='italic')
    
    def _draw_flattened_features(self, ax, count=3136, title=''):
        """Draw flattened features"""
        ax.axis('off')
        ax.set_xlim(-1, 11)
        ax.set_ylim(-0.5, 2.5)
        
        # Draw long horizontal bar representing flattened vector
        rect = Rectangle((0, 0), 10, 1,
                       facecolor=self.colors['flatten'], alpha=0.7,
                       edgecolor='black')
        ax.add_patch(rect)
        
        # Add some vertical lines to represent individual neurons
        for i in np.linspace(0.5, 9.5, 15):
            ax.plot([i, i], [0, 1], 'gray', alpha=0.3, linewidth=0.5)
        
        ax.text(5, 0.5, f"{count} neurons", ha='center', va='center',
               fontweight='bold', fontsize=9)
        ax.text(5, -0.3, title, ha='center', fontsize=8)
    
    def _draw_fc_layer(self, ax, neurons=128, title=''):
        """Draw a fully connected layer"""
        ax.axis('off')
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 3)
        
        # Draw neurons as circles
        for i in range(min(neurons, 12)):  # Show up to 12 neurons
            x = (i % 6) * 2 + 1
            y = 2 - (i // 6) * 2
            
            circle = Circle((x, y), 0.4,
                          facecolor=self.colors['fc'], alpha=0.7,
                          edgecolor='black')
            ax.add_patch(circle)
            
            # Add connections between some neurons
            if i > 0 and i < 5:
                ax.plot([x-2, x], [y, y], 'gray', alpha=0.2, linewidth=0.5)
        
        ax.text(5, -0.5, title, ha='center', fontsize=8)
        if neurons > 12:
            ax.text(5, 2.5, f"... +{neurons-12} more", ha='center', fontsize=7, style='italic')
    
    def _draw_output_layer(self, ax, title=''):
        """Draw output layer"""
        ax.axis('off')
        ax.set_xlim(-1, 11)
        ax.set_ylim(-0.5, 2.5)
        
        # Draw 10 output neurons (for digits 0-9)
        for i in range(10):
            x = i + 0.5
            circle = Circle((x, 1), 0.3,
                          facecolor=self.colors['output'], alpha=0.7,
                          edgecolor='black')
            ax.add_patch(circle)
            ax.text(x, 1, str(i), ha='center', va='center', fontsize=7, fontweight='bold')
        
        ax.text(5, -0.2, title, ha='center', fontsize=8)
        ax.text(5, 2.2, 'Digit Classes 0-9', ha='center', fontsize=7, style='italic')
    
    def _create_fc_description(self):
        """Create description for FC network"""
        return {
            'name': 'Fully Connected Network',
            'description': 'A traditional neural network where every neuron in one layer connects to every neuron in the next layer.',
            'architecture': [
                {'layer': 'Input', 'neurons': 784, 'activation': 'None', 'params': 0},
                {'layer': 'Hidden 1', 'neurons': 256, 'activation': 'ReLU', 'params': '200,960'},
                {'layer': 'Hidden 2', 'neurons': 128, 'activation': 'ReLU', 'params': '32,896'},
                {'layer': 'Hidden 3', 'neurons': 64, 'activation': 'ReLU', 'params': '8,256'},
                {'layer': 'Output', 'neurons': 10, 'activation': 'Softmax', 'params': '650'}
            ],
            'total_params': '242,762',
            'advantages': [
                'Simple to implement and understand',
                'Works well for small datasets',
                'Universal approximator (theoretically can learn any function)'
            ],
            'disadvantages': [
                'High parameter count leads to overfitting',
                'No spatial information preservation',
                'Computationally expensive for image data',
                'Poor scalability to large inputs'
            ],
            'use_cases': [
                'Small tabular datasets',
                'Simple classification tasks',
                'When interpretability is important'
            ]
        }
    
    def _create_cnn_description(self):
        """Create description for CNN network"""
        return {
            'name': 'Convolutional Neural Network',
            'description': 'A specialized neural network for processing grid-like data (images) using convolutional layers that preserve spatial relationships.',
            'architecture': [
                {'layer': 'Input', 'shape': '28×28×1', 'params': 0},
                {'layer': 'Conv2D (32 filters)', 'shape': '28×28×32', 'params': '320', 'kernel': '3×3'},
                {'layer': 'MaxPooling2D', 'shape': '14×14×32', 'params': 0, 'pool_size': '2×2'},
                {'layer': 'Conv2D (64 filters)', 'shape': '14×14×64', 'params': '18,496', 'kernel': '3×3'},
                {'layer': 'MaxPooling2D', 'shape': '7×7×64', 'params': 0, 'pool_size': '2×2'},
                {'layer': 'Flatten', 'shape': '3136', 'params': 0},
                {'layer': 'Dense', 'neurons': 128, 'params': '401,536', 'activation': 'ReLU'},
                {'layer': 'Output', 'neurons': 10, 'params': '1,290', 'activation': 'Softmax'}
            ],
            'total_params': '421,642',
            'key_concepts': [
                'Local Receptive Fields: Each neuron connects only to a small region of input',
                'Weight Sharing: Same filter applied across entire input',
                'Pooling: Downsampling to reduce spatial dimensions',
                'Hierarchical Feature Learning: Low-level → High-level features'
            ],
            'advantages': [
                'Parameter efficient (weight sharing)',
                'Preserves spatial information',
                'Translation invariant',
                'Excellent for image recognition',
                'Can handle variable input sizes'
            ],
            'disadvantages': [
                'More complex to implement',
                'Requires more data for training',
                'Computationally intensive during training',
                'Hyperparameter sensitive (kernel size, stride, padding)'
            ],
            'use_cases': [
                'Image classification',
                'Object detection',
                'Computer vision tasks',
                'Medical image analysis'
            ]
        }
    
    def generate_all_diagrams(self):
        """Generate all network diagrams"""
        print("Generating network architecture diagrams...")
        
        results = {}
        
        # Generate all diagrams
        results['fc_diagram'] = self.create_fc_network_diagram()
        results['cnn_diagram'] = self.create_cnn_network_diagram()
        results['comparison_diagram'] = self.create_comparison_diagram()
        results['feature_transformation'] = self.create_layer_wise_visualization()
        
        # Save descriptions to JSON
        descriptions = {
            'fc': self._create_fc_description(),
            'cnn': self._create_cnn_description()
        }
        
        with open(os.path.join(self.output_dir, 'network_descriptions.json'), 'w') as f:
            json.dump(descriptions, f, indent=2)
        
        print("✓ All diagrams generated successfully!")
        return results


# Flask integration helper
def integrate_with_flask(app):
    """Add routes for network visualization to Flask app"""
    
    @app.route('/network_diagrams')
    def show_network_diagrams():
        """Display network architecture diagrams"""
        # Generate diagrams if they don't exist
        visualizer = NetworkVisualizer()
        
        # Check if diagrams exist, generate if not
        diagrams_dir = 'static/network_diagrams'
        required_files = ['fc_network.png', 'cnn_network.png', 
                         'comparison_diagram.png', 'feature_transformation.png']
        
        if not all(os.path.exists(os.path.join(diagrams_dir, f)) for f in required_files):
            visualizer.generate_all_diagrams()
        
        # Load descriptions
        descriptions_file = os.path.join(diagrams_dir, 'network_descriptions.json')
        if os.path.exists(descriptions_file):
            with open(descriptions_file, 'r') as f:
                descriptions = json.load(f)
        else:
            descriptions = {
                'fc': visualizer._create_fc_description(),
                'cnn': visualizer._create_cnn_description()
            }
        
        return render_template('network_diagrams.html',
                             fc_desc=descriptions['fc'],
                             cnn_desc=descriptions['cnn'])
    
    @app.route('/generate_diagrams')
    def generate_diagrams():
        """Force regenerate all diagrams"""
        visualizer = NetworkVisualizer()
        results = visualizer.generate_all_diagrams()
        
        return jsonify({
            'success': True,
            'message': 'Diagrams generated successfully!',
            'diagrams': list(results.keys())
        })


# Standalone execution
if __name__ == "__main__":
    print("="*60)
    print("NETWORK ARCHITECTURE VISUALIZER")
    print("="*60)
    
    visualizer = NetworkVisualizer()
    
    # Generate all diagrams
    results = visualizer.generate_all_diagrams()
    
    print("\n📊 Generated Diagrams:")
    print(f"  • FC Network: {results['fc_diagram'][0]}")
    print(f"  • CNN Network: {results['cnn_diagram'][0]}")
    print(f"  • Comparison: {results['comparison_diagram']}")
    print(f"  • Feature Transformation: {results['feature_transformation']}")
    
    print("\n📋 Architecture Descriptions:")
    print(f"  • FC Parameters: {results['fc_diagram'][1]['total_params']}")
    print(f"  • CNN Parameters: {results['cnn_diagram'][1]['total_params']}")
    
    print("\n✅ Ready for web integration!")
    print("Add to Flask app with: integrate_with_flask(app)")