module write_module    
    #(  parameter DATA_WIDTH = 32,                                                  // The number of bits extracted from the BRAM per clock cycle
        parameter ADDR_WIDTH = 32,                                                  // The size of the BRAM addresses in bits
        parameter OUTPUT_ADDR = 32'hB000_2000,                                      // The base address of the BRAM for the kernel data
        parameter PIXEL_SIZE = 8,                                                   // Number of bits for 1 pixel   
        parameter PIXEL_PER_WORD = 4                                                // Number of pixels written at the same time to the BRAM
    )(     
        // System signals
        input logic clk,
        input logic reset,
        
        // BRAM signals
        output logic [ADDR_WIDTH-1:0] bram_addr,                                    // Only changing signals are the requested address, data to write and write enable
        output logic [DATA_WIDTH-1:0] bram_data,                                    // Others are assumed to stay constant
        output logic [3:0] write_enable,
        
        // Hardware Output
        input logic [PIXEL_SIZE-1:0] pixel,                                         // Output filter which has to be written in BRAM
        input logic pixel_valid,                                                    // Indication that the pixel signal is a valid output pixe
        input logic conv_done);                                                     // Convolution is done
        
        // Internal signals
        logic [ADDR_WIDTH-1:0] next_bram_addr;   
        logic [PIXEL_PER_WORD-1:0][PIXEL_SIZE-1:0] next_bram_data;                              
        logic [1:0]  index, next_index;                                              
        
        typedef enum {idle, collecting, writing} state_types;
        state_types state, next_state;
        
        // State registers
        always_ff @(posedge clk, posedge reset)
            if (reset) begin
                state <= idle;
            end
            else begin
                state <= next_state;
            end
            
        // Counter
        always_ff @(posedge clk, posedge reset)
            if (reset) begin
                index <= 3;
            end else begin
                index <= next_index;
            end 
            
        // Misc values
        always_ff @(posedge clk, posedge reset)
            if (reset) begin
                bram_addr <= OUTPUT_ADDR;
                bram_data <= 0;
            end else begin
                bram_addr <= next_bram_addr;
                bram_data <= next_bram_data;
            end
        
        always_comb begin
            write_enable = 4'b0000;
            next_bram_addr = bram_addr;
            next_bram_data = bram_data;
            next_state = state;
            next_index = index;
        
            unique case (state)
                idle: begin                                                             // Nothing is happening, automatically go to collect state due to nature pixel and pixel_valid combination
                    next_bram_addr = OUTPUT_ADDR;                                       // State only resets the values
                    next_bram_data = 0;
                    
                    next_index = 3;
                    
                    
                    next_state = collecting;
                end

                collecting: begin                                                       // Combine 8 bit valid pixels into a 32 bit word     
                    if (pixel_valid) begin                                              // Pixel valid? Make ready to write and increment index
                        next_index = index - 1;
                        next_bram_data[index] = pixel;
                    end
                    //write_enable = 4'b1111;
                    if (pixel_valid == 1 & index == 0) begin                                               // If word is complete go to writing
                        next_state = writing;
                    end 
                end
                
                writing: begin                                                          // Write words into BRAM, continue collecting pixels
                    if (pixel_valid) begin                                              // Pixel valid? Still make ready to write and increment index
                        next_index = index-1;
                        next_bram_data[index] = pixel;
                    end
                    
                    next_bram_addr = bram_addr + 4; 
                    write_enable = 4'b1111;                                             // Enable writing to the BRAM
                    
                    if (conv_done) begin                                                // Return to idle state when the convolution is done to reset values
                        next_state = idle;
                    end else begin
                        next_state = collecting;
                    end
                end
            endcase
        end                    
endmodule