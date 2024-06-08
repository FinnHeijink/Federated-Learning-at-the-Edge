module read_module
    #(  parameter DATA_WIDTH = 32,                                                  // The number of bits extracted from the BRAM per clock cycle
        parameter ADDR_WIDTH = 32,                                                  // The size of the BRAM addresses in bits
        parameter KERNEL_ADDR = 32'hB000_0000,                                      // The base address of the BRAM for the kernel data
        parameter IMAGE_ADDR = 32'hB000_0024,                                       // The base address of the BRAM for the image data    
        parameter PIXEL_SIZE = 8,                                                   // Number of bits for 1 pixel                
        parameter KERNEL_SIZE = 9,                                                  // Size of the kernel in terms of 32 bits!
        parameter KERNEL_WIDTH = 12,
        parameter IMAGE_SIZE = 784,  //196                                           // Size of the image in terms of 8 bits!
        parameter TOT_NUM_IMAGES = 4                                                // Number of images read in a batch
        //parameter SER_NUM_IMAGES                                                  // Number of images before the address has to be reset
    )(
        // System signals
        input logic clk,
        input logic reset,
        
        // BRAM signals
        output logic [ADDR_WIDTH-1:0] bram_addr,                                    // Only changing signals are the requested address and the output data
        input logic [DATA_WIDTH-1:0] bram_data,                                     // Others are assumed to stay constant

        // Kernel 
        input logic read_kernel,                                                    // Signal from PS, read the kernel
        output logic [KERNEL_SIZE-1:0][KERNEL_WIDTH-1:0] kernel,                    // Kernel register
        
        // Image
        input logic read_image,                                                     // Signal from PS, read the image
        output logic [PIXEL_SIZE-1:0] pixel,                                        // Pass along the read pixel to the filter
        output logic pixel_valid,                                                   // Indicate that a new pixel has been read
        
      
        input logic interrupt);                                                     // Wait with passing new pixels
        
        // Internal signals
        logic [ADDR_WIDTH-1:0] next_bram_addr;                                      
        logic [KERNEL_SIZE-1:0][DATA_WIDTH-1:0] next_kernel;
        logic prev_read_kernel, prev_read_image;                     
        logic [7:0]  kernel_count, next_kernel_count;
        logic [7:0]  image_count, next_image_count;
        logic [9:0]  pixel_count, next_pixel_count;
        logic [1:0]  index_count, next_index_count;
        
        typedef enum {idle, start_reading_kernel1, start_reading_kernel2, reading_kernel, start_reading_image1, start_reading_image2, reading_image, interruption} state_types;
        state_types state, next_state;                                              
        
        // State registers
        always_ff @(posedge clk, posedge reset)
            if (reset) begin
                state <= idle;
            end
            else if (interrupt) begin                                               // directly go to interruption state if an interruption occurs
                state <= interruption;
            end
            else begin
                state <= next_state;
            end
        
        // Counters
        always_ff @(posedge clk, posedge reset)
            if (reset) begin                                                        // reset all counters to 0
                kernel_count <= 0;
                pixel_count <= 0;
                image_count <= 0;
                index_count <= 0;
            end
            else if (interrupt) begin                                               // do not update any counters during an interruption
                kernel_count <= kernel_count;
                pixel_count <= pixel_count;
                image_count <= image_count;
                index_count <= next_index_count;
            end else begin
                kernel_count <= next_kernel_count;
                pixel_count <= next_pixel_count;
                image_count <= next_image_count;
                index_count <= next_index_count;
            end
        
        // Start signals to find rising edge
               always_ff @(posedge clk, posedge reset)
            if (reset) begin
                prev_read_kernel <= 0;
                prev_read_image <= 0;  
            end
            else begin
                prev_read_kernel <= read_kernel;
                prev_read_image <= read_image; 
            end 
        
        // Output values
        always_ff @(posedge clk, posedge reset)
            if (reset) begin
                bram_addr <= 0;                                           
                kernel <= 0;
            end
            else if (interrupt) begin                                               // do not update bram_address during an interruption
                bram_addr <= bram_addr;                                           
                kernel <= kernel; 
            end else begin
                bram_addr <= next_bram_addr;                                           
                kernel <= next_kernel;
            end
          
          
       always_comb begin
            // Standard assumed values stay equal
            next_bram_addr = bram_addr;
            next_kernel = kernel;
                   
            next_kernel_count = kernel_count;
            next_pixel_count = pixel_count;
            next_image_count = image_count;
            next_index_count = index_count;
                   
            next_state = state;
            unique case (state)
                idle: begin                                                         // Nothing is happening
                    pixel = 1'b0;                                                   // No pixel is being read, pixel is not valid
                    pixel_valid = 1'b0;
                   
                    next_kernel_count = 0;                                          // Counters should be reset in this state
                    next_pixel_count  = 0;
                    next_image_count  = 0;   
                   
                    if ((read_kernel == 1) && (prev_read_kernel == 0)) begin        // Go to start_reading_kernel on "rising edge" read_kernel to make preparations
                        next_state = start_reading_kernel1;
                    end else if ((read_image == 1) && (prev_read_image == 0)) begin // Go to start_reading_image on "rising edge" read_image to make preparations
                        next_state = start_reading_image1;
                    end
                end
                
                start_reading_kernel1: begin                                         // Make preparations to start reading the kernel, 2 cycles delay due to latency of BRAM
                    pixel = 1'b0;                                                   // No pixel is being read, pixel is not valid
                    pixel_valid = 1'b0;
                    
                    next_bram_addr = KERNEL_ADDR;   	                            // Already start adding to base address due to additional latency at BRAM side
                    
                    next_kernel_count = 0;                                          // Nothing has happened yet, so counters still zero
                    next_pixel_count  = 0;
                    next_image_count  = 0;  
                    next_index_count  = 0;  
                    
                    next_state = start_reading_kernel2;
                end
                
                start_reading_kernel2: begin                                         // Make preparations to start reading the kernel, 2 cycles delay due to latency of BRAM
                    pixel = 1'b0;                                                   // No pixel is being read, pixel is not valid
                    pixel_valid = 1'b0;
                    
                    next_bram_addr = KERNEL_ADDR + 4;                                   // Already start adding to base address due to additional latency at BRAM side
                    
                    next_kernel_count = 0;                                          // Nothing has happened yet, so counters still zero
                    next_pixel_count  = 0;
                    next_image_count  = 0;  
                    next_index_count  = 0;  
                    
                    next_state = reading_kernel;
                end
                
                reading_kernel: begin                                               // Read the kernel
                    pixel = 1'b0;                                                   // No pixel is being read, pixel is not valid
                    pixel_valid = 1'b0;
                    
                    next_bram_addr = bram_addr + 4;                                 // Increment address                                   
                    next_kernel[kernel_count] = bram_data[31:20];                   // Relevant index gets exchanged by data from BRAM, others stay the same
                    next_kernel[kernel_count+1] = bram_data[15:4];
                    
                    next_kernel_count = kernel_count + 2;                           // Increment index of the kernel register
                    next_pixel_count  = 0;                                          // Image related counters stay zero
                    next_image_count  = 0;    
                    next_index_count  = 0;
                    
                    if (kernel_count == (KERNEL_SIZE - 1)) begin
                        next_state = idle;                                          // Return to idle state when the kernel has been entirely read
                    end else begin
                        next_state = reading_kernel;
                    end
                end
                
                start_reading_image1: begin                                          // Make preparations to read the images
                    pixel = 1'b0;                                                   // No pixel is being read yet, pixel is not valid
                    pixel_valid = 1'b0;
                    
                    next_bram_addr = IMAGE_ADDR;                                    // Already start adding to base address due to additional latency at BRAM side, change value when num of pixels is increased     
                    
                    next_kernel_count = 0;                                          // Nothing has happened yet, so counters still zero
                    next_pixel_count  = 0;
                    next_image_count  = 0;    
                    
                    next_state = start_reading_image2;                              // Read images
                end
                
                start_reading_image2: begin                                          // Make preparations to read the images
                    pixel = 1'b0;                                                   // No pixel is being read yet, pixel is not valid
                    pixel_valid = 1'b0;
                    
                    next_bram_addr = IMAGE_ADDR;                                    // Already start adding to base address due to additional latency at BRAM side, change value when num of pixels is increased     
                    
                    next_kernel_count = 0;                                          // Nothing has happened yet, so counters still zero
                    next_pixel_count  = 0;
                    next_image_count  = 0;    
                    
                    next_state = reading_image;                                     // Read images
                end
                
                reading_image: begin                                                // Read images, change state when num of pixels has been changed
                    case (index_count) 
                        0: begin pixel = bram_data[31:24];  end
                        1: begin pixel = bram_data[23:16];  end
                        2: begin pixel = bram_data[15:8];  end
                        3: begin pixel = bram_data[7:0];  end
                    endcase
                    pixel_valid = 1'b1;                                             // Pixel is valid!!
                    
                    if (index_count == 2) begin
                        next_bram_addr = bram_addr + 4;                             // Incrementation timed based on latency, TEST THIS
                    end
                    
                    next_kernel_count = 0;                                          // Kernel counter stays zero
                    next_index_count = index_count + 1;
                    if (pixel_count == (IMAGE_SIZE -1)) begin                       // Image is done? Increment image counter and reset pixel counter
                        next_image_count = image_count + 1;                         
                        next_pixel_count = 0;                            
                    end else begin
                        next_pixel_count  = pixel_count + 1;
                    end
                    
                    if (image_count == (TOT_NUM_IMAGES)) begin                      // All images are done? Go back to idle
                        next_state = idle;                                          
                    end else begin
                        next_state = reading_image;
                    end
                end
                
                interruption: begin                                                 // filter needs more time, all values need to stay equal, TEST THIS
                    pixel = 0;                                                      // 
                    pixel_valid = 1'b0;                                             // Pixel is not valid!!
                    
                    if (!interrupt) begin                                           // interruption done? Go back to reading
                        next_state = reading_image;
                        if (index_count == 3) begin                                 // start updating index count and bram_addr to ensure correct order of pixels
                            next_bram_addr = bram_addr + 4;
                            next_index_count = index_count;
                        end
                    end else begin
                        next_state = interruption;
                    end
                end
            endcase
        end
endmodule
