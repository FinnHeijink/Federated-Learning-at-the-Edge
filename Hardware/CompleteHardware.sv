module CompleteHardware
    #(
        ADDR_WIDTH_PS = 32,                                     // Addr width of BRAMs connected to PS
        DATA_WIDTH_PS = 32,                                     // Data width of data from PS connected BRAMs
        ADDR_WIDTH_INT = 11,                                    // Addr width of augmentation BRAMs
        PIXEL_WIDTH = 8,                                        // size of one pixel
        NUM_IMAGES = 16,                                        // number of images in a batch
        KERNEL_SIZE = 9,                                        // number of total kernel weights
        KERNEL_WIDTH = 12,                                      // number of bits per kernel weight
        NUM_PIXELS = 784,                                       // number of pixels in an image
        BRAM_INT_BASE1 = 11'h000,                               // base address of the first image in the augmentation BRAMs
        BRAM_INT_BASE2 = 11'h310,                               // base address of the second image in the augmentation BRAMs
        IMAGE_BASE_ADDR = 32'h0000_0024                         // base address of the images in the BRAM buffer at the start 
    )
    (
        // system signals
        input logic clk,
        input logic reset,
        
        // control signals
        input logic read_kernel,
        input logic start,
        
        // interface with start buffer
        output logic [ADDR_WIDTH_PS-1:0]    bram_buffer_addr,
        input logic  [DATA_WIDTH_PS-1:0]     bram_buffer_data,
        
        // interface blur bram
        output logic [ADDR_WIDTH_INT-1:0]   blur_bram_addr_in,                       
        output logic [PIXEL_WIDTH-1:0]      blur_bram_data_in,                    
        output logic                        blur_bram_w_enable,                        
        output logic [ADDR_WIDTH_INT-1:0]   blur_bram_addr_out,                      
        input logic  [PIXEL_WIDTH-1:0]       blur_bram_data_out,
    
        // interface rot bram
        output logic [ADDR_WIDTH_INT-1:0]   rot_bram_addr_in,                          
        output logic [PIXEL_WIDTH-1:0]      rot_bram_data_in,                           
        output logic                        rot_bram_w_enable,                          
        output logic [ADDR_WIDTH_INT-1:0]   rot_bram_addr_out,                          
        input logic  [PIXEL_WIDTH-1:0]       rot_bram_data_out,
        
        //interface intermediate results
        output logic [ADDR_WIDTH_PS-1:0]    bram_int_results_addr,
        output logic [DATA_WIDTH_PS-1:0]    bram_int_results_data,
        output logic [3:0]                  bram_int_results_w_enable,
        
        //interface end results
        output logic [ADDR_WIDTH_PS-1:0]    bram_end_results_addr,
        output logic [DATA_WIDTH_PS-1:0]    bram_end_results_data,
        output logic [3:0]                  bram_end_results_w_enable
    );
    
        // internal signals
        logic [KERNEL_SIZE-1:0][KERNEL_WIDTH-1:0]   kernel;
        logic [PIXEL_WIDTH-1:0] pixel_read, pixel_augment, pixel_convolved;
        logic pixel_read_valid, pixel_augment_valid, pixel_convolved_valid;
        logic interrupt;

                
        // port maps
        read_module
        #(.IMAGE_ADDR(IMAGE_BASE_ADDR), 
          .KERNEL_SIZE(KERNEL_SIZE),                                                  
          .TOT_NUM_IMAGES(1))
          read                                                                                          
         (.clk(clk),
          .reset(reset),
          .bram_addr(bram_buffer_addr),                                    
          .bram_data(bram_buffer_data),                                     
          .read_kernel(read_kernel),                                                    
          .kernel(kernel),                    
          .read_image(start),                                                     
          .pixel(pixel_read),                                        
          .pixel_valid(pixel_read_valid),                                                   
          .interrupt(interrupt));  
          
        Augmentation
        #()
        augment
         (.clk(clk),
          .reset(reset),
          .pixel_in(pixel_read),                                 
          .pixel_in_valid(pixel_read_valid),                            
          .read_start(int_start),                             
          .blur_bram_addr_in(blur_bram_addr_in),                       
          .blur_bram_data_in(blur_bram_data_in),                    
          .blur_bram_w_enable(blur_bram_w_enable),                        
          .blur_bram_addr_out(blur_bram_addr_out),                      
          .blur_bram_data_out(blur_bram_data_out),                       
          .rot_bram_addr_in(rot_bram_addr_in),                          
          .rot_bram_data_in(rot_bram_data_in),                           
          .rot_bram_w_enable(rot_bram_w_enable),                          
          .rot_bram_addr_out(rot_bram_addr_out),                          
          .rot_bram_data_out(rot_bram_data_out),                         
          .pixel_out(pixel_augment),                                  
          .pixel_out_valid(pixel_augment_valid),                             
          .interrupt(interrupt));
          
        write_module    
        #()
        write_intermediate
        (.clk(clk),
         .reset(reset),
         .bram_addr(bram_int_results_addr),                                    
         .bram_data(bram_int_results_data),                                   
         .write_enable(bram_int_results_w_enable),
         .pixel(pixel_augment),                                       
         .pixel_valid(pixel_augment_valid));  
         
        write_module    
        #()
        write_end
        (.clk(clk),
         .reset(reset),
         .bram_addr(bram_end_results_addr),                                    
         .bram_data(bram_end_results_data),                                   
         .write_enable(bram_end_results_w_enable),
         .pixel(pixel_convolved),                                       
         .pixel_valid(pixel_convolved_valid)); 
endmodule