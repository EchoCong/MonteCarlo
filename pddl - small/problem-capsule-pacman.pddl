(define (problem pacman-problem)
   (:domain pacman)
   (:objects p_0_0 p_0_1 p_0_2 p_0_3 p_0_4 p_1_0 p_1_1 p_1_2 p_1_3 p_1_4 p_2_0 p_2_1 p_2_2 p_2_3 p_2_4 p_3_0 p_3_1 p_3_2 p_3_3 p_3_4 p_4_0 p_4_1 p_4_2 p_4_3 p_4_4 - pos)
   (:init 
   ;; Grid 5x5
   ;; P - Pacman, G - Ghost, # - wall, F - Food, C - Capsule
   ;; |P| |#|#|#|
   ;; | | | | | |
   ;; |#|#|#| | |
   ;; |F|G|G| |C|
   ;; |#|#|F| | |
   ;; In this case, there are two foods, one of them is surrunded by walls and ghosts, 
   ;; the pacman is supposed to eat capsule first, then it can eat the  
   ;; food surrunded by ghosts.
        (unPowered)
		(At p_0_0)
		(FoodAt p_3_0)
		(FoodAt p_4_2)
		(GhostAt p_3_1)
		(GhostAt p_3_2)
		(CapsuleAt p_3_4)
		(Adjacent p_0_0 p_0_1)
		(Adjacent p_0_1 p_0_0)
		(Adjacent p_0_0 p_1_0)
		(Adjacent p_1_0 p_0_0)
		(Adjacent p_0_1 p_1_1)
		(Adjacent p_1_1 p_0_1)
		(Adjacent p_1_0 p_1_1)
		(Adjacent p_1_1 p_1_0)
		(Adjacent p_1_1 p_1_2)
		(Adjacent p_1_2 p_1_1)
		(Adjacent p_1_2 p_1_3)
		(Adjacent p_1_3 p_1_2)
		(Adjacent p_1_3 p_1_4)
		(Adjacent p_1_4 p_1_3)
		(Adjacent p_1_3 p_2_3)
		(Adjacent p_2_3 p_1_3)
		(Adjacent p_1_4 p_2_4)
		(Adjacent p_2_4 p_1_4)
		(Adjacent p_2_3 p_2_4)
		(Adjacent p_2_4 p_2_3)
		(Adjacent p_2_3 p_3_3)
		(Adjacent p_3_3 p_2_3)
		(Adjacent p_2_4 p_3_4)
		(Adjacent p_3_4 p_2_4)
		(Adjacent p_3_0 p_3_1)
		(Adjacent p_3_1 p_3_0)
		(Adjacent p_3_1 p_3_2)
		(Adjacent p_3_2 p_3_1)
		(Adjacent p_3_2 p_3_3)
		(Adjacent p_3_3 p_3_2)
		(Adjacent p_3_3 p_3_4)
		(Adjacent p_3_4 p_3_3)
		(Adjacent p_3_2 p_4_2)
		(Adjacent p_4_2 p_3_2)
		(Adjacent p_3_3 p_4_3)
		(Adjacent p_4_3 p_3_3)
		(Adjacent p_3_4 p_4_4)
		(Adjacent p_4_4 p_3_4)
		(Adjacent p_4_2 p_4_3)
		(Adjacent p_4_3 p_4_2)
		(Adjacent p_4_3 p_4_4)
		(Adjacent p_4_4 p_4_3)
    )
   (:goal 
   (and 
	    (not 
	        (and 
	           (CapsuleAt p_3_4)
	           (FoodAt p_4_2)
	        	 (FoodAt p_3_0)
	        ))
	    (At p_0_0)
	)
    )
)