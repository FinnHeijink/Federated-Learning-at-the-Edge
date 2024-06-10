module AugmentationTestKRIA
    #(
        ADDR_WIDTH_PS = 32,                                     // Addr width of BRAMs connected to PS
        DATA_WIDTH_PS = 32,                                     // Data width of data from PS connected BRAMs
        ADDR_WIDTH_INT = 11,                                    // Addr width of augmentation BRAMs
        PIXEL_WIDTH = 8,                                        // size of one pixel
        NUM_IMAGES = 8,                                        // number of images in a batch
        KERNEL_SIZE = 9,                                        // number of total kernel weights
        KERNEL_WIDTH = 12,                                      // number of bits per kernel weight
        NUM_PIXELS = 784,                                       // number of pixels in an image
        BRAM_INT_BASE1 = 11'h000,                               // base address of the first image in the augmentation BRAMs
        BRAM_INT_BASE2 = 11'h310,                               // base address of the second image in the augmentation BRAMs
        IMAGE_BASE_ADDR = 32'h0000_0024                         // base address of the images in the BRAM buffer at the start 
    )();
    
        // system signals
        logic clk;
        logic reset;
        
        // control signals
        logic read_kernel;
        logic start;
        
        // interface with start buffer
        logic [ADDR_WIDTH_PS-1:0]    bram_buffer_addr;
        logic [DATA_WIDTH_PS-1:0]     bram_buffer_data;
        
        // interface blur bram
        logic [ADDR_WIDTH_INT-1:0]   blur_bram_addr_in;                       
        logic [PIXEL_WIDTH-1:0]      blur_bram_data_in;                    
        logic                        blur_bram_w_enable;                        
        logic [ADDR_WIDTH_INT-1:0]   blur_bram_addr_out;                      
        logic [PIXEL_WIDTH-1:0]       blur_bram_data_out;
    
        // interface rot bram
        logic [ADDR_WIDTH_INT-1:0]   rot_bram_addr_in;                          
        logic [PIXEL_WIDTH-1:0]      rot_bram_data_in;                           
        logic                        rot_bram_w_enable;                          
        logic [ADDR_WIDTH_INT-1:0]   rot_bram_addr_out;                          
        logic [PIXEL_WIDTH-1:0]       rot_bram_data_out;
        
        //interface intermediate results
        logic [ADDR_WIDTH_PS-1:0]    bram_int_results_addr;
        logic [DATA_WIDTH_PS-1:0]    bram_int_results_data;
        logic [3:0]                  bram_int_results_w_enable;   

        logic [7:0]                  not_used, not_used2;
        logic [31:0]                    not_used1;
        
        // port maps
        AugmentationTest(.*);
        
        ResizedCropBS_wrapper
       (.blur_bram_in_addr(blur_bram_addr_in),
        .blur_bram_in_clk(clk),
        .blur_bram_in_din(blur_bram_data_in),
        .blur_bram_in_dout(not_used),
        .blur_bram_in_en(1'b1),
        .blur_bram_in_we(blur_bram_w_enable),
        .blur_bram_out_addr(blur_bram_addr_out),
        .blur_bram_out_clk(clk),
        .blur_bram_out_din(8'b0),
        .blur_bram_out_dout(blur_bram_data_out),
        .blur_bram_out_en(1'b1),
        .blur_bram_out_we(1'b0),
        .bram_input_addr(bram_buffer_addr),
        .bram_input_clk(clk),
        .bram_input_din(32'b0),
        .bram_input_dout(bram_buffer_data),
        .bram_input_en(1'b1),
        .bram_input_rst(1'b0),
        .bram_input_we(4'b0),
        .bram_output_addr(bram_int_results_addr),
        .bram_output_clk(clk),
        .bram_output_din(bram_int_results_data),
        .bram_output_dout(not_used1),
        .bram_output_en(1'b1),
        .bram_output_rst(1'b0),
        .bram_output_we(bram_int_results_w_enable),
        .clock(clk),
        .restart_tri_o(reset),
        .read_kernel_tri_o(read_kernel),
        .rot_bram_in_addr(rot_bram_addr_in),
        .rot_bram_in_clk(clk),
        .rot_bram_in_din(rot_bram_data_in),
        .rot_bram_in_dout(not_used2),
        .rot_bram_in_en(1'b1),
        .rot_bram_in_we(rot_bram_w_enable),
        .rot_bram_out_addr(rot_bram_addr_out),
        .rot_bram_out_clk(clk),
        .rot_bram_out_din(8'b0),
        .rot_bram_out_dout(rot_bram_data_out),
        .rot_bram_out_en(1'b1),
        .rot_bram_out_we(1'b0),
        .start_tri_o(start));

endmodule