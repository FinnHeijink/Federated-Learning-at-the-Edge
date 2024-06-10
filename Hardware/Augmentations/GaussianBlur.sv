module GaussianBlur                                                         // Apply a Gaussian Blur (or does not) and writes into the internal bram memory
    #(
        PIXEL_WIDTH = 8,                                                    // size of one pixel
        NUM_PIXELS = 784,                                                   // number of pixels in an image
        KERNEL_DIM = 3,                                                     // kernel of dimension x dimension                                    
        KERNEL_WIDTH = 12,                                                  // the width of a weight in the kernel
        ADDR_WIDTH = 11                                                     // the address width of the BRAM                                     
    )
    (
        // system signals
        input logic                     clk,
        input logic                     reset,
        
        // input signals
        input logic [7:0]               pixel_in,                           // pixel read from buffer
        input logic                     pixel_in_valid,                     // the read pixel is valid
        
        // singals to BRAM
        output logic [ADDR_WIDTH-1:0]   bram_addr,                          // bram address to which is written
        output logic [PIXEL_WIDTH-1:0]  bram_data,                          // data written to bram
        output logic                    w_enable,                           // write enable of bram
        
        // control signals
        output logic                    interrupt,                          // read module should wait until the CLA is done before sending new pixels
        output logic                    image_done                          // an image is totally augmented and written to memory 
    );
    

    
        // internal signals
        logic                           LFSR_update;                        // value of the LFSR should change
        logic [1:0]                     select;                             // selects whether an Gaussian Blur is applied and which one
        logic                           blur_active, skip_active;           // pixel_in_valid is only high for one of the two options modules 
        logic                           skip_done, blur_done;               // a module has processed an entire image
        logic                           skip_written, blur_written;         // a module has processed an entire image
        logic [KERNEL_DIM-1:0][KERNEL_DIM-1:0][KERNEL_WIDTH-1:0] kernel;    // kernel for the CLA
        logic [PIXEL_WIDTH-1:0]         pixel_blur, pixel_skip;             // output pixels of the modules
        logic                           skip_valid, blur_valid;             // this pixel is valid
        logic [ADDR_WIDTH-1:0]          blur_addr, skip_addr;               // address to the bram for the two different modules
        logic [PIXEL_WIDTH-1:0]         blur_data, skip_data;               // data to the bram for the two different modules
        logic                           blur_w_enable, skip_w_enable;       // write enable to the bram for the two different modules
        logic                           CLA_interrupt, delay_interrupt;     // read module should wait until the CLA is up to date or the delay between augmenting and writing is gone      
        logic                           reset_CLA;                           // CLA is reset after an image is read or external reset is used.
        
        assign LFSR_update = skip_written | blur_written;                   // update the value after all values have been written
        assign image_done = skip_done | blur_done;                          // image is done augmented when either of the two modules is done
        assign interrupt = CLA_interrupt | delay_interrupt;                  // read module is interrupted when the CLA needs a stall or when an image is done ugmented but not written yet
        assign reset_CLA = reset | blur_written;                            // CLA has to be resetted by hand

        typedef enum {idle, delay} state_types;
        state_types state, next_state;

        always_ff@(posedge(clk), posedge(reset))
        if (reset)
            state <= idle;
        else
            state <= next_state;
            
        always_comb begin
            delay_interrupt = 0;
            case (state) 
                idle: begin
                    if (skip_done ==1 | blur_done ==1) begin                // cause an interruption when skip or blur is done but is not written completely yet
                        next_state = delay;
                        delay_interrupt = 1;
                    end else next_state = idle;
                end
                delay: begin
                    delay_interrupt = 1;
                    if (skip_written ==1 | blur_written==1) begin
                        next_state = idle;
                    end else next_state = delay;
                end
                default: begin
                    next_state = idle;
                    delay_interrupt = 0;              
                end
            endcase
        end

        // multiplexer set up
        always_comb begin
            case (select)
                2'b00: begin                                               // skip module used
                    skip_active = pixel_in_valid;
                    blur_active = 0;
                    kernel = 0; 
                    bram_addr = skip_addr;
                    bram_data = skip_data;
                    w_enable = skip_w_enable;
                    end
                2'b01: begin                                               // skip module used
                    skip_active = pixel_in_valid;
                    blur_active = 0;
                    kernel = 0; 
                    bram_addr = skip_addr;
                    bram_data = skip_data;
                    w_enable = skip_w_enable;
                    end
                2'b10: begin                                               // skip module used
                    skip_active = pixel_in_valid;
                    blur_active = 0;
                    kernel = 0; 
                    bram_addr = skip_addr;
                    bram_data = skip_data;
                    w_enable = skip_w_enable;
                    end
                2'b11: begin                                               // image is blurred
                    skip_active = 0;
                    blur_active = pixel_in_valid;
                    kernel[0] = {12'd1, 12'd2,12'd1};
                    kernel[1] = {12'd2, 12'd4, 12'd2};
                    kernel[2] = {12'd1, 12'd2,12'd1}; 
                    bram_addr = blur_addr;
                    bram_data = blur_data;
                    w_enable = blur_w_enable;
                    end
                default: begin
                    skip_active = 0;
                    blur_active = 0;
                    kernel = 0; 
                    bram_addr = 0;
                    bram_data = 0;
                    w_enable = 0;
                end
            endcase
        end        
        
        // portmaps
        LFSR_blur L1                                                        // Generate a pseudo-random value to determine whether a gaussian blur is applied
       (.clk(clk),
        .reset(reset),
        .enable(LFSR_update),
        .select(select)
       );
        
        top_conv 
       (.clk_i(clk),
        .rst_ni(reset_CLA),
        .k_val(kernel),
        .pixel_i(pixel_in),
        .pix_data_valid(blur_active),
        .pixel_o(pixel_blur),
        .conv_finished(blur_valid),
        .finished_image(image_done),
        .read_all_pixels(CLA_interrupt)
);

        write_blur L3
       (.clk(clk),
        .reset(reset),
        .bram_addr(blur_addr),
        .bram_data(blur_data),                                    
        .w_enable(blur_w_enable),
        .pixel(pixel_blur),                                         
        .pixel_valid(blur_valid),                                   
        .image_done(blur_written)); 
        
        SkipBlur L4 
       (.pixel_in(pixel_in),
        .pixel_in_valid(skip_active),
        .pixel_out(pixel_skip),
        .pixel_out_valid(skip_valid),
        .clk(clk),
        .reset(reset),
        .image_done(skip_done));
        
        write_augmented L5
       (.clk(clk),
        .reset(reset),
        .bram_addr(skip_addr),
        .bram_data(skip_data),                                    
        .w_enable(skip_w_enable),
        .pixel(pixel_skip),                                         
        .pixel_valid(skip_valid),                                   
        .image_done(skip_written));
endmodule