module filter_fsm #(
  parameter width = 12,
  parameter input_width = 8,
  parameter im_dim = 28,
  parameter k_size = 9, // 3x3
  parameter k_dim = 3 // 3x3 kernel
  )(
  input  logic                  clk_i,
  input  logic                  reset,
  
  input logic [input_width-1:0] pixel_i,
    
  input logic                   pix_data_valid,
  
  output logic           [7:0]  pixel_o,
  
  output logic                  conv_finished,
    
  output logic					finished_image,
  
  output logic					read_all_pixels,
  
  input logic [107:0] k_val
);

  typedef enum {
      init, 
      await_buffers,
      process, 
	  stall_conv
  } state_t; 
    
  state_t state_d, state_q; 
  
  typedef enum {
    idle, 
    convolve,
	stall_reading
  } read_state;

  read_state r_state_d, r_state_q;
  
  // logic [2:0][2:0][input_width-1:0] pixel_i;
  logic [2:0][2:0][input_width-1:0] pixel_grid;
    
  logic [1:0] conv_stall; 
  
  logic reset_buffers;
  
  logic read_buffer;
  
  logic conv_data_valid;
  logic [1:0] increment_buffer;
  
  logic [$clog2(im_dim)-1:0] row_pix_counter; // 5 bit signal
  logic [$clog2(im_dim*im_dim)-1:0] pix_counter; // 10 bit signal (28^2)
  
  logic [1:0] active_write_buffer;
  logic [1:0] active_read_buffer;
  
  logic p1_valid, p2_valid, p3_valid, p4_valid;
  logic [3:0] buff_valid;
  logic [4:0] read_counter;
  
  logic [2:0][input_width-1:0] p1_buffer, p2_buffer, p3_buffer, p4_buffer;
  
  logic [$clog2(im_dim)-1:0] read_row_counter; // Tracks total number of read rows committed to buffers, to know when an image is entirely read!
    
  always_ff @(posedge clk_i or posedge reset) begin
    if (reset) begin
      state_q <= init;  
      r_state_q <= idle; 
    end else begin
      state_q <= state_d;
      r_state_q <= r_state_d;
    end 
  end
  
  always_ff @(posedge clk_i or posedge reset) begin
    if (reset || reset_buffers) begin
      pix_counter <= '0;  
    end else if (pix_data_valid) begin
      pix_counter <= pix_counter + 1'b1;
    end 
  end
  
  // Counts the # of pixel in 'active row'
  always_ff @(posedge clk_i or posedge reset) begin
    if (reset || reset_buffers) begin
      row_pix_counter <= '0;  
    end else if (pix_data_valid && row_pix_counter == im_dim-1) begin
      row_pix_counter <= '0;
    end else if (pix_data_valid) begin
      row_pix_counter <= row_pix_counter + 1'b1;
    end 
  end
  
  always_ff @(posedge clk_i or posedge reset) begin
    if (reset || reset_buffers) begin
	  conv_stall <= 1'b0;
	end else if (read_buffer && read_counter == im_dim-3 && conv_stall == 'd0) begin
	  conv_stall <= 2'b10; // conv_stall = 2
	end else if (conv_stall > 'd0) begin
	  conv_stall <= conv_stall - 1'b1;
	end 
  end 
  
  always_ff @(posedge clk_i or posedge reset) begin
    if (reset || reset_buffers) begin 
	  increment_buffer <= '0;
	end else if (read_buffer && read_counter == im_dim-3 && conv_stall == 'd0) begin 
	  increment_buffer <= 2'b01; // increment_buffer = 1
	end else if (conv_stall > 'd0) begin
	  if (increment_buffer == 2'b10) begin 
	    increment_buffer <= 1'b0;
	  end else begin 
	    increment_buffer <= increment_buffer + 1'b1;
	  end 
	end 
  end 
  
  always_ff @(posedge clk_i or posedge reset) begin
    if (reset || reset_buffers) begin
      read_all_pixels <= 1'b0;
    end else if (pix_counter >= 'd783) begin
      read_all_pixels <= 1'b1;
    end
  end

  // Tracks the row buffer that corresponds with the given row being written to ('active write row')
  always_ff @(posedge clk_i or posedge reset) begin
    if (reset || reset_buffers) begin
      active_write_buffer <= '0;  // start with p1
    end else if (row_pix_counter == im_dim-1 && pix_data_valid) begin
      active_write_buffer <= active_write_buffer + 1'b1; // exploit overflow to loop back to 0!
    end
  end

  always_comb begin
    buff_valid = '0;
    buff_valid[active_write_buffer] = pix_data_valid; // Validity signal determines which row buffer stores input pixel
	{p4_valid, p3_valid, p2_valid, p1_valid} = buff_valid;
  end

  always_ff @(posedge clk_i or posedge reset) begin
    if (reset || reset_buffers) begin
      read_counter <= '0;  
      read_row_counter <= '0;
    end else if (read_buffer && read_counter == im_dim-3 && conv_stall == 'd0) begin
      read_counter <= '0;
      read_row_counter <= read_row_counter + 1'b1; 
    end else if (read_buffer && conv_stall == 'd0) begin
      read_counter <= read_counter + 1'b1; // counts every row during active convolution phase, similar to row_pix_counter
	  read_row_counter <= read_row_counter; 
    end
  end
  

  always_ff @(posedge clk_i or posedge reset) begin
    if (reset || reset_buffers) begin
      active_read_buffer <= '0;
    end else if (increment_buffer == 2'b10) begin
      active_read_buffer <= active_read_buffer + 1;  
    end
  end

  always_comb begin
    case(active_read_buffer)
      2'b00: begin
        pixel_grid = {p1_buffer, p2_buffer, p3_buffer};
      end
      2'b01: begin
        pixel_grid = {p2_buffer, p3_buffer, p4_buffer};
      end
      2'b10: begin
        pixel_grid = {p3_buffer, p4_buffer, p1_buffer};
      end
      2'b11: begin
        pixel_grid = {p4_buffer, p1_buffer, p2_buffer};
      end
      default: begin
        pixel_grid = '0;
      end
    endcase
  end
  
  always_comb begin 
    if ((r_state_q == stall_reading || r_state_q == convolve) && r_state_d == idle) begin
	  reset_buffers <= 1'b1;
	end else begin 
	  reset_buffers <= 1'b0; 
	end 
  end 

  always_comb begin
    case(r_state_q) // @suppress "Default clause missing from case statement"
      idle: begin
	    read_buffer <= 1'b0;
        if (pix_counter == (im_dim*k_dim-1)) begin // active_write_buffer ought to be at state 2'b11
          r_state_d <= convolve; // First 3 lines have been read 
        end else begin
          r_state_d <= idle;
        end
      end 
      
      convolve: begin
        if (read_row_counter == im_dim-2) begin
		  read_buffer <= 1'b0; 
          r_state_d <= idle;
        end else if (conv_stall > 'd0) begin 
		  read_buffer <= 1'b1; 
		  r_state_d <= stall_reading;
		end else begin 
		  read_buffer <= 1'b1; 
          r_state_d <= convolve;
        end
      end
	  
	  stall_reading: begin 
	    read_buffer <= 1'b0;
		if (read_row_counter == im_dim-2) begin
		  r_state_d <= idle; 		
		end else if (conv_stall == 'd0) begin 
		  r_state_d <= convolve;
		end else begin 
		  r_state_d <= stall_reading;
		end 
	  end
    endcase
  end

  always_comb begin
    case(state_q)
      init: begin
        conv_data_valid <= 1'b0;
		finished_image <= 1'b0; 
        state_d <= await_buffers; 
      end
       
      await_buffers: begin
        conv_data_valid <= 1'b0;
		finished_image <= 1'b0; 
        if (pix_counter == im_dim*k_dim-1) begin
          state_d <= process;
        end else begin
          state_d <= await_buffers;
        end
      end

      process: begin
        if (read_row_counter == im_dim-2 && conv_finished == 1'b1) begin
		  finished_image <= 1'b1; 
		  conv_data_valid <= 1'b0;
          state_d <= await_buffers;
        end else if (conv_stall > 'd0) begin
		  finished_image <= 1'b0; 
		  conv_data_valid <= 1'b0;
		  state_d <= stall_conv;
		end else begin
		  finished_image <= 1'b0; 
		  conv_data_valid <= 1'b1;
          state_d <= process;
        end
      end
	  
	  stall_conv: begin
		conv_data_valid <= 1'b0;  
		if (read_row_counter == im_dim-2 && conv_finished == 1'b1) begin 
		  finished_image <= 1'b1;
		  state_d <= await_buffers;  
		end else if (conv_stall == 'd0) begin
		  finished_image <= 1'b0;
		  state_d <= process;
		end else begin
		  finished_image <= 1'b0;
          state_d <= stall_conv;
		end 
	  end

    endcase
  end

  // pipelined filter!
  grayscale_conv conv_layer (
    .clk_i(clk_i), 
    .reset(reset), 
    .pixel_i(pixel_grid),
    .data_valid(conv_data_valid),
    .k_i(k_val),
    .pixel_o(pixel_o),
    .output_valid(conv_finished)  
  );
 
  // The following "pixel buffers" function as row buffers, buffering row-data for the kernel
  
  pixel_buffer # (
	.input_width(input_width),
	.buff_size(k_dim),
	.im_dim(im_dim)
  ) p1 (
    .clk_i(clk_i),
    .reset(reset),
    .pix_data(pixel_i),
    .data_valid(p1_valid),
	.read_address(read_counter),
    .pix_buffer(p1_buffer)  
  );
  
  pixel_buffer # (
	.input_width(input_width),
	.buff_size(k_dim),
	.im_dim(im_dim)
  ) p2 (
    .clk_i(clk_i),
    .reset(reset),
    .pix_data(pixel_i),
    .data_valid(p2_valid),
	.read_address(read_counter),
    .pix_buffer(p2_buffer)  
  );
  
  pixel_buffer # (
	.input_width(input_width),
	.buff_size(k_dim),
	.im_dim(im_dim)
  ) p3 (
    .clk_i(clk_i),
    .reset(reset),
    .pix_data(pixel_i),
    .data_valid(p3_valid),
	.read_address(read_counter),
    .pix_buffer(p3_buffer)
  );
  
  pixel_buffer # (
	.input_width(input_width),
	.buff_size(k_dim),
	.im_dim(im_dim)
  ) p4 (
    .clk_i(clk_i),
    .reset(reset),
    .pix_data(pixel_i),
    .data_valid(p4_valid),
	.read_address(read_counter),
    .pix_buffer(p4_buffer)  
  );
    
endmodule : filter_fsm