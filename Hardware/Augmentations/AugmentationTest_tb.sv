module AugmentationTest_tb
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
        logic [DATA_WIDTH_PS-1:0]    bram_buffer_data;
        
        // interface blur bram
        logic [ADDR_WIDTH_INT-1:0]   blur_bram_addr_in;                       
        logic [PIXEL_WIDTH-1:0]      blur_bram_data_in;                    
        logic                        blur_bram_w_enable;                        
        logic [ADDR_WIDTH_INT-1:0]   blur_bram_addr_out;                      
        logic [PIXEL_WIDTH-1:0]      blur_bram_data_out;
    
        // interface rot bram
        logic [ADDR_WIDTH_INT-1:0]   rot_bram_addr_in;                          
        logic [PIXEL_WIDTH-1:0]      rot_bram_data_in;                           
        logic                        rot_bram_w_enable;                          
        logic [ADDR_WIDTH_INT-1:0]   rot_bram_addr_out;                          
        logic [PIXEL_WIDTH-1:0]      rot_bram_data_out;
        
        //interface intermediate results
        logic [ADDR_WIDTH_PS-1:0]    bram_int_results_addr;
        logic [DATA_WIDTH_PS-1:0]    bram_int_results_data;
        logic [3:0]                  bram_int_results_w_enable;
        
        //interface end results
        logic [ADDR_WIDTH_PS-1:0]    bram_end_results_addr;
        logic [DATA_WIDTH_PS-1:0]    bram_end_results_data;
        logic [3:0]                  bram_end_results_w_enable;
        
        AugmentationTest test1(.*);
        BRAM_sim_biiiig test2(bram_buffer_addr,bram_buffer_data,clk,reset) 
        
        always
            #5 clk = ~clk;  // period 10ns (100 MHz)
        initial
            clk = 0;
    
        initial begin
        reset = 1; start = 0;   
        #20;       reset = 0; 
        #20;       start = 1;
        #20;       start = 0;     
    end  
endmodule    