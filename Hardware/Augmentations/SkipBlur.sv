module SkipBlur                                                 // passes a pixel stream along
    #(  
        PIXEL_WIDTH = 8,                                        // size of one pixel
        NUM_PIXELS = 784                                        // number of pixels in an image
    )
    (
        // system signals
        input logic                     clk,
        input logic                     reset,
        
        // pixel stream in 
        input logic [PIXEL_WIDTH-1:0]   pixel_in,
        input logic                     pixel_in_valid,
        
        // pixel stream out
        output logic [PIXEL_WIDTH-1:0]  pixel_out,
        output logic                    pixel_out_valid,

        output logic                    image_done
    );
    

        logic [9:0] pixel_count, next_pixel_count;              // keeps track of the number of pixels
    
        always_ff @(posedge clk, posedge reset)
        if (reset) begin
            pixel_count <= 0;
        end else begin
            pixel_count <= next_pixel_count;
        end
        
        always_comb begin
            if (pixel_in_valid == 1) begin                      // if a pixel is valid, pixel counter is incremented
                if (pixel_count == NUM_PIXELS-1) begin
                    next_pixel_count = 0;
                    image_done = 1;                             // after an entire image, a done signal is asserted
                end else begin
                next_pixel_count = pixel_count + 1;
                image_done = 0;
                end
            end else begin                                      // if a pixel is not valid, values keep their value
                next_pixel_count = pixel_count;
                image_done = 0;
            end
            pixel_out_valid = pixel_in_valid;                   // values are copied to output
            pixel_out = pixel_in;
        end
            
endmodule