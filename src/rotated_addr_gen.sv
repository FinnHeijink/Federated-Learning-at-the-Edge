module rotated_addr_gen #(
  parameter theta = 30
  )(
  input logic clk_i,
  input logic rst_ni,
  input logic [7:0] pixel_i,
  output logic [9:0] rot_addr,
  output logic [7:0] pixel_r,
  output logic pix_valid
 );
  // State encoding
  typedef enum logic [1:0] {
      IDLE,
      CALC,
      FETCH,
	  DONE
  } state_t;

  state_t state_q, state_d;
	
  logic signed [12:0] a, b; // a = cos(theta), b = sin(theta)
  
  // --- Theta = 30 deg ---
  assign a = 13'd3547; // cos(theta) << 12
  assign b = 13'd2048; // sin(theta) << 12
  
  logic signed [4:0] x_rot, y_rot, x_loc, y_loc, x, y;
  
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (~rst_ni) begin
	  state_q <= IDLE; 
	  rot_addr <= '0;
	  pixel_r <= '0; 
	  pix_valid <= 1'b0;
	  x_rot <= x_rot;
	  y_rot <= y_rot;
	end else begin 
	  state_q <= state_d; 
	  case (state_q)   
	    IDLE: begin
		  pixel_r <= '0; 
		  pix_valid <= 1'b0;
		end 
		
		CALC: begin
		  x_loc <= x_rot - 5'd14;
		  y_loc <= 5'd14 - y_rot; 
		  x <= $signed(a*x_loc + b*y_loc) >> 12; 
		  y <= $signed(-b*x_loc + a*y_loc) >> 12;
		  pixel_r <= '0;
		  pix_valid <= 1'b0;
		  if ((x >= -14) && (x <= 14) && (y >= -14) && (y <= 14)) begin
			rot_addr <= {5'd14 - y, 5'd14 + x};
          end else begin 
		    rot_addr <= '0; 
		  end 
		end 
		
		FETCH: begin
		  pixel_r <= (rot_addr == 10'b0) ? 8'b0 : pixel_i;
		  pix_valid <= 1'b1;
		  if (x_rot == 'd27) begin 
		    x_rot <= '0;
			y_rot <= y_rot + 1'b1; 
		  end else begin 
			x_rot <= x_rot + 1'b1;
		  end 
		end 

		
		DONE: begin
		  pixel_r <= '0; 
		  pix_valid <= 1'b0; 
		end 
	  endcase 
	end 
  end 
  
  always_comb begin // State-transition logic!
    case (state_q)
	  IDLE: begin 
	    state_d <= CALC; 
	  end 
	  
	  CALC: begin
	    state_d <= FETCH;
	  end 
	  
	  FETCH: begin
	    if (y_rot == 'd27 && x_rot == 'd27) begin
		  state_d <= DONE;
		end else begin 
	      state_d <= CALC; 
		end 
	  end 
	  
	  DONE: begin
	    state_d <= state_q;
	  end  
	endcase 
  end 
  
  
endmodule : rotated_addr_gen