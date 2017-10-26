(define (problem ghost-problem)
   (:domain ghost)
   (:objects  
   t_10 t_9 t_8 t_7 t_6 t_5 t_4 t_3 t_2 t_1 t_0 
   
   -time
   
   p_0_0 p_0_1 p_0_2 p_0_3 p_0_4 p_1_0 p_1_1 p_1_2 p_1_3 
   p_1_4 p_2_0 p_2_1 p_2_2 p_2_3 p_2_4 p_3_0 p_3_1 p_3_2 p_3_3
   p_3_4 p_4_0 p_4_1 p_4_2 p_4_3 p_4_4
   
   - pos
   )
   
   (:init 
   ;; Grid 5x5
   ;; P - Pacman, G - Ghost, # - wall, F - Food, C - Capsule
   ;;
   ;; | | |#|#|#|
   ;; | | | | | |
   ;; |#|#|#| | |
   ;; |P| | |G| |
   ;; |#|#| | |P|
   ;; In this case, the ghost was not sacared (the ScaredTime is set to t_0)
   ;; The ghost is supposed to go for pacman once the program starts
   
		(At p_3_3)       ;; ghost initial position
		(PacmanAt p_3_0) ;; target postion
		(PacmanAt p_4_4)
		(ScaredTime t_0)
        (NotScaredTime t_0)

		(TimeCounter t_10 t_9)
		(TimeCounter t_9 t_8)
		(TimeCounter t_8 t_7)
		(TimeCounter t_7 t_6)
		(TimeCounter t_6 t_5)
		(TimeCounter t_5 t_4)
		(TimeCounter t_4 t_3)
		(TimeCounter t_3 t_2)
		(TimeCounter t_2 t_1)
		(TimeCounter t_1 t_0)
		
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
   (:goal (and (not (PacmanAt p_3_0))
	           (not (PacmanAt p_4_4))
	       )
    )
)